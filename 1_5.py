#异构图：异构图里可以有不同类型的节点和边
import dgl
import torch

#创建具有3中节点类型和3种边类型的异构图
data_graph={
    #dgl.heterograph({('node_type', 'edge_type', 'node_type'): (u, v)})  这种格式
    #dgl.heterograph({('source_type', 'edge_type', 'destination_type'): (u, v)})
    ('drug','interacts','drug'):(torch.tensor([0,1]),torch.tensor([1,2])),
    ('drug','interacts','gene'):(torch.tensor([0,1]),torch.tensor((2,3))),
    ('drug', 'treats', 'disease'):(torch.tensor([1]),torch.tensor([2]))
}
g=dgl.heterograph(data_graph)    #创建一个异构图对象
print(g)                  #与异构图相关联的 metagraph 就是图的模式
print(g.ntypes)
print(g.etypes)
print(g.canonical_etypes)
#当引入多种节点和边类型后，用户在调用DGLGraph API以获取特定类型的信息时，需要指定具体的节点和边类型。此外，不同类型的节点和边具有单独的ID。
print(g.num_nodes())
#获取drug节点的数量
print(g.num_nodes("drug"))
print(g.nodes("drug"))
print(g.nodes("gene"))
print(g.nodes("disease"))
#为某一类型的节点和边设置特征属性
g.nodes['drug'].data['hv'] = torch.ones(3, 1)
g.edges['treats'].data['he']=torch.zeros(1,1)
print(g.edges['treats'].data['he']) #当边类型唯一地确定了源节点和目标节点的类型时，用户可以只使用一个字符串而不是字符串三元组来指定边类型。
#对于'interacts'则要用三元组形式避免歧义
g.edges[('drug','interacts','drug')].data['hee']=torch.zeros(g.num_edges(('drug','interacts','drug')),1)
print(g.edges[('drug','interacts','drug')].data['hee'])

#提供了dgl.savegraphs()和dgl.loadgraphs()来保存和加载图(二进制格式保存)
#取异构子图
eg = dgl.edge_type_subgraph(g, [('drug', 'interacts', 'drug'),
                                ('drug', 'treats', 'disease')])
print(eg)


#将异构图转化为同构图
g = dgl.heterograph({
   ('drug', 'interacts', 'drug'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
   ('drug', 'treats', 'disease'): (torch.tensor([1]), torch.tensor([2]))})
g.nodes['drug'].data['hv'] = torch.zeros(3, 1)
g.nodes['disease'].data['hv'] = torch.ones(3, 1)
g.edges['interacts'].data['he'] = torch.zeros(2, 1)
g.edges['treats'].data['he'] = torch.zeros(1, 2)
hg = dgl.to_homogeneous(g)
print(hg)
hg = dgl.to_homogeneous(g, edata=['he']) #边的’he‘特征维度不匹配
hg = dgl.to_homogeneous(g,ndata=['hv'])