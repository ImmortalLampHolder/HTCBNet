import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
from nets.base import DWConv1d,MSDWConv,ConvFFN,SEModule,get_relative_distances,MLP


class LGMSA(nn.Module):
    def __init__(self, dim, heads_num, window_size, seq_len, relative_pos_embedding=True):
        super(LGMSA, self).__init__()
        # shared
        self.head_dim = dim // heads_num
        self.heads_num = heads_num
        self.inner_dim = self.head_dim * self.heads_num
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        self.relative_pos_embedding = relative_pos_embedding

        self.to_v = nn.Linear(dim, self.inner_dim)
        self.to_out = nn.Linear(self.inner_dim, dim)

        # local branch
        if self.relative_pos_embedding:
            self.w_relative_indices = get_relative_distances(window_size) + window_size - 1
            self.w_pos_embedding = nn.Parameter(torch.randn(2*window_size - 1))
        else:
            self.w_pos_embedding = nn.Parameter(torch.randn(window_size, window_size))

        self.to_wqk = nn.Linear(dim, self.inner_dim*2)
        self.sign1 = nn.Identity()
        # gobal branch
        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(seq_len) + seq_len - 1
            self.pos_embedding = nn.Parameter(torch.randn(2*seq_len - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(seq_len, seq_len))

        self.to_qk = nn.Linear(dim, self.inner_dim*2)
        self.sign2 = nn.Identity()

    def forward(self,x):
        b, L, _, h = *x.shape, self.heads_num
        wqk = self.to_wqk(x).chunk(2, dim=-1)
        n = L // self.window_size
        v = self.to_v(x)

        # local branch
        wv = rearrange(v, 'b (n ws) (h d) -> b h (n) (ws) d',h=h, ws=self.window_size)
        wq, wk = map(
            lambda t: rearrange(t, 'b (n ws) (h d) -> b h (n) (ws) d',
                                h=h, ws=self.window_size), wqk)
        wdots = torch.einsum('b h n i d, b h w j d -> b h n i j', wq, wk) * self.scale
        if self.relative_pos_embedding:
            wdots += self.w_pos_embedding[self.w_relative_indices[:, 0]]
        else:
            wdots += self.w_pos_embedding

        wattn = wdots.softmax(dim=-1)
        wattn = self.sign1(wattn)
        wout = torch.einsum('b h w i j, b h w j d -> b h w i d', wattn, wv)
        wout = rearrange(wout, 'b h n ws d -> b (n ws) (h d)',
                        h=h, n=n, ws=self.window_size)

        # gobal branch
        v = rearrange(v, 'b l (h d) -> b h l d', h=h)
        qk = self.to_qk(x).chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=h),qk)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, 0]]
        else:
            dots += self.pos_embedding
        attn = dots.softmax(dim=-1)
        attn = self.sign2(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h l d -> b l (h d)', h=h)
        return self.to_out(wout + out)


class MSConv(nn.Module):
    def __init__(self, dim):
        super(MSConv, self).__init__()
        self.dw = nn.Conv1d(dim, dim, 5, 1, 2, groups=dim, bias=False)
        self.ms = MSDWConv(dim)
        self.se = SEModule(dim)
        self.ffn = ConvFFN(dim, dim*4)

    def forward(self,x):
        x = self.dw(x) + x
        x = self.ffn(self.se(self.ms(x))) + x
        return x


class LGFormer(nn.Module):
    def __init__(self,dim, heads_num, window_size, seq_len):
        super(LGFormer, self).__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.msa = LGMSA(dim, heads_num, window_size, seq_len)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim*4, dim)

    def forward(self,x):
        x = self.msa(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x
        return x

# net = LGFormer(32,4,50,400)
# data = torch.randn(1,400,32)
# out = net(data)
# print(out.shape)




