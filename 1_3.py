#对象的节点和边可具有多个用户定义的、可命名的特征，以储存图的节点和边的属性，ndate和edate
import dgl
import torch

g=dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0]))
g.ndata['x']=torch.ones(g.num_nodes(),3)   #图的'x'属性，每个属性由三位表示
#print(g.ndata['x'])
g.edata['x']=torch.ones(g.num_edges(),dtype=torch.int32)
print(g)
print(g.edata['x'])

g.ndata['y']=torch.rand(g.num_nodes(),5)
print(g.ndata['y'])
print(g.edata['x'][1])    #获取第1个边的‘x’特征

print(g.ndata['y'][torch.tensor([1,4])])  #获取节点1和4的‘y’特征

#对于加权图，可以将权重作为一个特征保存在图中
weight = torch.tensor([5,8,1,5],dtype=torch.float32)
g.edata['weight']=weight


