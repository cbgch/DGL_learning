import torch as t
from torch import nn
import numpy as np
# in_features由输入张量的形状决定，out_features则决定了输出张量的形状
connected_layer = nn.Linear(in_features = 64*64*3, out_features = 1)

# 假定输入的图像形状为[64,64,3]
input = t.randn(3,64,64,3)

# 将四维张量转换为二维张量之后，才能作为全连接层的输入
input = input.view(3,64*64*3)
print(input.shape)
output = connected_layer(input) # 调用全连接层
print(output.shape)

list=[1,2,3,4,5,6,7,8,9]
print(list[1:-1])

print(np.random.permutation(2708))
# print("************************************")
nodes_batch=[5,1,56,41,4,8,2,7,]
nodes_batch_layers = [(nodes_batch,)]
print(nodes_batch_layers)
print(type(nodes_batch_layers))
list = np.array(nodes_batch)
print(list.shape)
# print("************************************")
_set = set
print(type(_set))
print("************************************")
_unique_nodes_list=[1,2,3,4]
i=[5,6,7,8]
#li=list(zip(_unique_nodes_list, i))
li=zip(_unique_nodes_list, i)
print(li)
print(type(li))
ret=dict(li)
print(ret)
print(len(ret))