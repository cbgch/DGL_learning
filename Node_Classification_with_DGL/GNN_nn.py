import dgl
import torch.nn as nn
import torch
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv


class GCN(nn.Module):
    #in_feats --> 1443     h_feats --> 16       num_classes --> 7
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
