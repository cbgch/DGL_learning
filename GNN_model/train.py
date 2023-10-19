import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

def train(g,model):
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    all_logits=[]
    best_val_acc=0
    best_test_acc=0

    feats=g.ndata['feat']
    labels=g.ndata['label']
    train_mask=g.ndata['train_mask']
    val_mask=g.ndata['val_mask']
    test_mask=g.ndata['test_mask']

    for e in range(301):
        logits=model(g,feats)

        pred=logits.argmax(1)
        loss=F.cross_entropy(logits[train_mask],labels[train_mask])

        train_acc=(pred[train_mask]==labels[train_mask]).float().mean()
        val_acc=(pred[val_mask]==labels[val_mask]).float().mean()
        test_acc=(pred[test_mask]==labels[test_mask]).float().mean()

        if val_acc>best_val_acc:
            best_test_acc=test_acc
            best_val_acc=val_acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_logits.append(logits.detach())  #detach()函数将两个list从内存中分离,不需要更新梯度

        if(e%5==0):
            print(
                "In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})".format(
                    e, loss, val_acc, best_val_acc, test_acc, best_test_acc
                )
            )



