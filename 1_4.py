#从外部源创建图
import dgl
import torch
import scipy.sparse as sp
import networkx as nx

matr=sp.rand(100,100,density=0.05,dtype=int)    #density表示生成矩阵的密度
#print(type(matr))
g=dgl.from_scipy(matr)
#print(g)

#networkx创建的是无向图
nx_g=nx.path_graph(5)
print(type(nx_g))
g2=dgl.from_networkx(nx_g)
print(g2)

nxg = nx.DiGraph([(1,2),(2,1),(2,3),(0,0)])   #使用networkx创建有向图
g3=dgl.from_networkx(nxg)
print(g3)