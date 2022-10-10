import torch.nn as nn
from dgl.nn.pytorch import GATConv
from contrast import Contrast
import torch.nn.functional as F

class MVHGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,tau, lam):
        super(MVHGAT,self).__init__()
        nheads=8
        self.dropout=dropout
        self.gat1 = GATConv(nfeat, nhid, num_heads=nheads)
        # self.gat2 = GATConv(nhid, nhid, num_heads=num_heads)
        self.gat2=GATConv(nhid*nheads, nhid, 1)
        if dropout > 0:
            self.feat_drop = nn.Dropout(dropout)
        else:
            self.feat_drop = lambda x: x

        self.constast=Contrast(nhid, tau, lam)


    def forward(self,gp, gf,feat_p,pos):

        # 前面使用拼接，后面平均
        support1 = self.gat1(gp,feat_p).flatten(1) #[4019,8,64],加了flatten之后为[4019,512]
        support1 = self.feat_drop(support1)

        outputp = self.gat2(gp,support1).mean(1)
        # outputp=F.log_softmax(outputp, dim=1)


        support2 = self.gat1(gf, feat_p).flatten(1)
        support2 = self.feat_drop(support2)
        outputf =self.gat2(gf,support2).mean(1)
        # outputf = F.log_softmax(outputf, dim=1)

        loss = self.constast(outputp,outputf,pos)

        return loss

    def get_embeds(self,gp,gf,  feat_p):
        support1 = self.gat1(gp, feat_p).flatten(1)  # [4019,8,64]
        support1 = self.feat_drop(support1)
        outputp = self.gat2(gp, support1).mean(1)

        support2 = self.gat1(gf, feat_p).flatten(1)
        support2 = self.feat_drop(support2)
        outputf = self.gat2(gf, support2).mean(1)
        return outputp.detach()