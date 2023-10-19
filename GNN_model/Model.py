from SAGEConv import SAGEConv
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self,in_feats,h_feats,num_labels):
        super(Model, self).__init__()
        self.layer_1=SAGEConv(in_feats,h_feats)
        self.layer_2=SAGEConv(h_feats,num_labels)

    def forward(self,g,in_feats):
        h=self.layer_1(g,in_feats)
        h=F.relu(h)
        h=self.layer_2(g,h)
        return h

