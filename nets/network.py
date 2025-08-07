import torch
import torch.nn as nn
from nets.base import Conv1d,ResNetBlock,Transformer
from nets.modules import MSConv,LGFormer


class Stem(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Stem, self).__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.cv = Conv1d(in_dim,out_dim,k=9,s=1)

    def forward(self,x):
        x = self.bn(x)
        return self.cv(x)


class Local(nn.Module):
    def __init__(self,dim, n=6):
        super(Local, self).__init__()
        self.block = nn.Sequential(
            *[ResNetBlock(dim, k=9) for _ in range(n)]
        )

    def forward(self,x):
        return self.block(x)


class Global(nn.Module):
    def __init__(self, dim, n=6, hn=4, ws=50, sq=400):
        super(Global, self).__init__()
        self.block = nn.Sequential(
            *[LGFormer(dim, hn, ws, sq) for _ in range(n)]
        )

    def forward(self,x):
        return self.block(x.transpose(1, 2)).transpose(1, 2)


class Fusion(nn.Module):
    def __init__(self,dim):
        super(Fusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(dim*2, dim, 9, 1, 4),
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, dim, 9, 1, 4),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 9, 1, 4),
            nn.Sigmoid()
        )

    def forward(self,lx,gx):
        w_lx = self.gate(torch.cat([lx, gx], dim = 1))
        w_gx = torch.ones_like(w_lx) - w_lx
        return w_lx * lx + w_gx * gx


class JRCPredictor(nn.Module):
    def __init__(self, dim, hidden_len=800):
        super(JRCPredictor, self).__init__()
        self.cv = nn.Conv1d(dim, 1, 1, 1,bias=False)
        self.fc = nn.Linear(hidden_len,1)
        self.act = nn.Sigmoid()

    def forward(self,x):
        x = self.cv(x)
        x = self.fc(x)
        x = self.act(x)
        return x


class HTCBNet(nn.Module):
    def __init__(self,in_dim=1, base_dim=32,n=6,ws=50,sq=400):
        super(HTCBNet, self).__init__()
        self.stem  = Stem(in_dim,base_dim)
        self.local_block = Local(base_dim,n)

        self.global_block = Global(base_dim,n,4,ws,sq)

        self.fusion = Fusion(base_dim)
        self.feature = nn.Identity()
        self.bigru  = nn.GRU(sq, sq, 1, batch_first=True, bidirectional=True)
        self.jrc_proj = JRCPredictor(base_dim, sq*2)

    def forward(self,x):
        x = self.stem(x)
        gx = self.global_block(x)
        lx = self.local_block(x)
        x = self.fusion.forward(lx, gx)
        x = self.feature(x)
        if not hasattr(self, '_flattened'):
            self.bigru.flatten_parameters()
            setattr(self, '_flattened', True)
        x, _ = self.bigru(x)
        x = self.jrc_proj(x)
        return x