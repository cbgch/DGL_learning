#使用dgl.graph 创建DGLGraph对象
import dgl
import torch

u=torch.tensor([0,0,0,1])   #源节点编号
v=torch.tensor([1,2,3,3])   #torch.Tensor()中的数据是float类型的，torch.tensor()中的数据根据类型推断
dglGraph=dgl.graph((u,v))   #创建的图如果存在孤立的节点，则需要指明创建节点的数量 num_nodes=
print(dglGraph)

#获取图中所有节点的id
print(dglGraph.nodes())
#获取图中所有的边
print(dglGraph.edges())
#获取图中所有的边及其对应的id
print(dglGraph.edges(form="all"))

#创建无向图的方法
dglGraph=dgl.to_bidirected(dglGraph)
print(dglGraph.edges(form="all"))

#查看节点id的数据类型
print(dglGraph.idtype)
dglGraph=dglGraph.int()  #转化为32位的整型
print(dglGraph.idtype)