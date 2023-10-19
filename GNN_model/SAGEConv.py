import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    DGL在dgl.nn.SAGEConv中内置了GraphSAGE的实现
"""

class SAGEConv(nn.Module):
    def __init__(self,in_feat,out_feat):
        super(SAGEConv, self).__init__()
        self.linear=nn.Linear(in_features=2*in_feat,out_features=out_feat)

    def forward(self,g,h):
        with g.local_scope():
            g.ndata['h']=h;
            g.update_all(
                message_func=fn.copy_u('h','m'),
                reduce_func=fn.mean('m','h_N'),
            )
            h_N=g.ndata['h_N']
            total=torch.cat([h,h_N],dim=1)
            return self.linear(total)

