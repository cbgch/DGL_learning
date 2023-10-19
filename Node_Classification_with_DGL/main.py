"""
    GCN进行节点分类
"""
import os
from GNN_nn import GCN
from train import train
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F

dataset = dgl.data.CoraGraphDataset()
print(f"Number of categories: {dataset.num_classes}")

print("**********************!!!!!!!!!****************************")

g = dataset[0]     #一个DGL数据集可能含有多张图(cora只有一张图)
print(g)
print("Node features")
print(g.ndata)
print(g.ndata['label'])
print("Edge features")
print(g.edata)


# Create the model with given dimensions
model = GCN(g.ndata["feat"].shape[1], 16, dataset.num_classes)  #g.ndata["feat"].shape[1] --> 1433

train(g, model)


