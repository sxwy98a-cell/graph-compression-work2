from typing import Union

from torch import Tensor
from torch_sparse import SparseTensor
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSage(torch.nn.Module):
    # def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
    #              with_relu=True, with_bias=True, with_bn=False, device=None):
    def __init__(self
                 , nfeat
                 , nhid
                 , nclass
                 , nlayers
                 , dropout=0.5
                 , batchnorm=False,
                weight_decay=5e-4,
                 with_relu=True, with_bias=True, with_bn=False, device=None):
        super(GraphSage, self).__init__()
        self.with_bn = batchnorm
        self.layers = torch.nn.ModuleList()
        self.layers.append(SAGEConv(nfeat, nhid))
        self.bns = torch.nn.ModuleList()

        if self.with_bn:
            self.bns.append(torch.nn.BatchNorm1d(nhid))
        for _ in range(nlayers - 2):
            self.layers.append(SAGEConv(nhid, nhid))
            if self.with_bn:
                self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.layers.append(SAGEConv(nhid, nclass))

        self.dropout = dropout
    def initialize(self):

        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index: Union[Tensor, SparseTensor]):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            output = x
            if self.with_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return output, x.log_softmax(dim=-1)
