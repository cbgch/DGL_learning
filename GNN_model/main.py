from Model import Model
import train
import dgl.data


dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

model = Model(g.ndata["feat"].shape[1], 16, dataset.num_classes)   #g.ndata["feat"].shape[1] --> 2708
train.train(g,model)