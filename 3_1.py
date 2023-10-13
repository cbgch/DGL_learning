import torch.nn as nn
from dgl.utils import expand_as_pair #Return a pair of same element if the input is not a pair.

class SAGEconv(nn.module):
    # DGL NN模块的构造函数
    # 1.设置选项 2.设置可学习的参数和模块 3.初始化参数
    def __init__(self,
                 in_feat,
                 out_feat,
                 aggregator_type,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEconv,self).__init__()

        self.in_src_feats,self.in_out_feats = expand_as_pair(in_feat)
        self.out_feats=out_feat
        self.aggr_type=aggregator_type
        self.norm=norm
        self.activation=activation
        if aggregator_type not in ["mean","gcn","pool","lstm"]:
            raise KeyError(f"aggregate funtion is wrong")
        if aggregator_type == "pool":
            self.fc_pool=nn.Linear(self.in_src_feats,self.in_src_feats)
        if aggregator_type == "lstm":
            self.lstm=nn.LSTM(self.in_src_feats,self.in_src_feats,batch_first=True)
        if aggregator_type in ["mean","pool","lstm"]:
            self.fc_self=nn.Linear(self.in_src_feats,self.out_feats,bias=bias)
        self.fc_neigh=nn.Linear(self.in_src_feats,self.out_feats,bias=bias)
        self.reset_parameter()       #进行权重初始化

    def reset_parameter(self):
        """重新初始化可学习的参数"""
        gain=nn.init.calculate_gain("relu")
        if self.aggr_type=="pool":
            nn.init.xavier_uniform_(self.fc_pool.weight,gain=gain)
        if self.aggr_type=="lstm":
            self.lstm.reset_parameters()
        if self.aggr_type!="gcn":
            nn.init.xavier_uniform_(self.fc_self.weight,gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight,gain=gain)


    #forward函数执行了实际的消息传递和计算
    """
        检测输入图对象是否符合规范。
        消息传递和聚合。
        聚合后，更新特征作为输出。
    """
    def forward(self,g,feat):
        with g.local_scope():
            feat_src,feat_dst = expand_as_pair(feat,g)


