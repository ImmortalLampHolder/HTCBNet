import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np


class Conv1d(nn.Module):
    def __init__(self,c1,c2,k,s,g=1,bias = False,act = True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=c1, out_channels=c2, kernel_size=k, stride=s, padding=k//2, groups=g, bias=bias)
        self.bn    = nn.BatchNorm1d(c2)
        self.act   = nn.ReLU() if act else nn.Identity()

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        return self.act(x)


class DWConv1d(Conv1d):
    def __init__(self,c1,c2,k,s,bias = False,act = True):
        super(DWConv1d, self).__init__(c1,c2,k,s,math.gcd(c1,c2),bias,act)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()

        # Project the input into query, key, and value vectors
        Q = self.query(x)  # (batch_size, seq_length, embed_dim)
        K = self.key(x)  # (batch_size, seq_length, embed_dim)
        V = self.value(x)  # (batch_size, seq_length, embed_dim)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,
                                                                                    2)  # (batch_size, num_heads, seq_length, head_dim)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,
                                                                                    2)  # (batch_size, num_heads, seq_length, head_dim)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,
                                                                                    2)  # (batch_size, num_heads, seq_length, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_length, seq_length)
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_length, seq_length)

        # Apply attention weights to the values
        out = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_length, head_dim)

        # Concatenate the heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length,
                                                    embed_dim)  # (batch_size, seq_length, embed_dim)

        # Final linear layer
        out = self.out(out)  # (batch_size, seq_length, embed_dim)

        return out


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class Transformer(nn.Module):
    def __init__(self,embed_dim, num_heads):
        super(Transformer, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.MHSA = MultiHeadAttention(embed_dim,num_heads)
        self.mlp  = MLP(embed_dim,embed_dim*4,embed_dim)

    def forward(self,x):
        short = x
        x = self.ln1(x)
        x = self.MHSA(x) + short
        short = x
        x = self.ln2(x)
        x = self.mlp(x) + short
        return x


class ResNetBlock(nn.Module):
    def __init__(self,dim,k=51):
        super(ResNetBlock, self).__init__()
        c = dim // 2
        self.cv1 = Conv1d(dim,c,k=1,s=1)
        self.cv2 = Conv1d(c,c,k=k,s=1)
        self.cv3 = Conv1d(c,dim,k=1,s=1)

    def forward(self,x):
        short = x
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.cv3(x)
        return x + short


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts = self.displacement, dims= 1)


class ConvFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            Conv1d(dim, hidden_dim, 1, 1, act=False),
            nn.GELU(),
            Conv1d(hidden_dim, dim, 1, 1, act=False),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement):
    mask = torch.zeros(window_size, window_size)
    mask[-displacement:, :-displacement] = float('-inf')
    mask[:-displacement, -displacement:] = float('-inf')
    return mask


def get_relative_distances(window_size):
    indices  = torch.from_numpy(np.arange(window_size,dtype=np.int32))
    distances = indices[None, :] - indices[:, None]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2*window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size, window_size))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)
        b, L, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        n = L // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (n ws) (h d) -> b h (n) (ws) d',
                                h=h, ws=self.window_size), qkv)

        dots = torch.einsum('b h n i d, b h w j d -> b h n i j', q, k) * self.scale
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:,0]]
        else:
            dots += self.pos_embedding
        if self.shifted:
            dots[:, :, -n:] += self.mask
        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h n ws d -> b (n ws) (h d)',
                        h=h, n=n, ws=self.window_size)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class MSDWConv(nn.Module):
    def __init__(self, dim):
        super(MSDWConv, self).__init__()
        self.dw1 = DWConv1d(dim, dim, 5,  1, act=False)
        self.dw2 = DWConv1d(dim, dim, 11, 1, act=False)
        self.dw3 = DWConv1d(dim, dim, 19, 1, act=False)
        self.dw4 = DWConv1d(dim, dim, 31, 1, act=False)

    def forward(self,x):
        return self.dw1(x) + self.dw2(x) + self.dw3(x) + self.dw4(x)


class SEModule(nn.Module):
    def __init__(self, dim):
        super(SEModule, self).__init__()
        self.block = nn.Sequential(
            Conv1d(dim, dim//2, 1, 1),
            nn.Conv1d(dim//2, dim, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.block(x)


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attention_block = WindowAttention(dim=dim,heads=heads,
                                               head_dim=head_dim,shifted=shifted,
                                               window_size=window_size,
                                               relative_pos_embedding=relative_pos_embedding)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp_block = MLP(dim,mlp_dim,dim)

    def forward(self, x):
        x = self.attention_block(self.ln1(x)) + x
        x = self.mlp_block(self.ln2(x)) + x
        return x


class SwinStageBlock(nn.Module):
    def __init__(self,dim, heads, head_dim, mlp_dim, window_size, relative_pos_embedding=True):
        super(SwinStageBlock, self).__init__()
        self.wsa = SwinBlock(dim, heads, head_dim, mlp_dim, False, window_size, relative_pos_embedding)
        self.swsa = SwinBlock(dim, heads, head_dim, mlp_dim, True, window_size, relative_pos_embedding)

    def forward(self,x):
        x = self.wsa(x)
        x = self.swsa(x)
        return x
