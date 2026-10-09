from typing import Union

from torch import Tensor
from torch_sparse import SparseTensor
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv


class GAT(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, dropout,
                 layer_heads=[4, 2, 1], batchnorm=False, with_bias=False, device='cuda:1'):
        super(GAT, self).__init__()

        self.layers = torch.nn.ModuleList()
        # self.bns = torch.nn.ModuleList() if batchnorm else None
        self.batchnorm = batchnorm
        self.with_bias = with_bias
        self.dropout = dropout

        # 第一层
        self.layers.append(GATConv(nfeat, nhid, heads=layer_heads[0], concat=True))
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(nhid * layer_heads[0]))

        # 中间层 (核心修改点)
        for i in range(1, nlayers - 1):
            self.layers.append(
                GATConv(nhid * layer_heads[i - 1], nhid, heads=layer_heads[i], concat=True)
            )
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(nhid * layer_heads[i]))

        # 最后一层
        self.layers.append(GATConv(nhid * layer_heads[nlayers - 2], nclass, heads=layer_heads[-1], concat=False))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index: Union[Tensor, SparseTensor]):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            mid_x = x
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return mid_x, x.log_softmax(dim=-1)
    def initialize(self):
        """
           Initialize parameters of GCN.
        """
        for layer in self.layers:  # 对每层进行初始化操作
            layer.reset_parameters()
        if self.with_bias:
            for bn in self.bns:
                bn.reset_parameters()

class GATv2(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , layer_heads=[]
                 , batchnorm=True):
        super(GATv2, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=layer_heads[0], concat=True))
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * layer_heads[0]))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(hidden_channels * layer_heads[i - 1], hidden_channels, heads=layer_heads[i], concat=True))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels * layer_heads[i - 1]))
        self.convs.append(GATv2Conv(hidden_channels * layer_heads[num_layers - 2]
                                    , out_channels
                                    , heads=layer_heads[num_layers - 1]
                                    , concat=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index: Union[Tensor, SparseTensor]):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)