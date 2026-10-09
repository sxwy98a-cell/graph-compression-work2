from typing import Union
from torch.nn.parameter import Parameter
import math
import torch_sparse
from torch import Tensor
from torch_sparse import SparseTensor
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn.modules.module import Module


from sklearn import metrics
'''
此模型邻居矩阵采用sparse tensor的形式，可以大大减少计算量，
如果不使用sparse tensor形式传递，将adj_t替换成edge_index
'''
class GCN(torch.nn.Module):
    def __init__(self
                 , nfeat
                 , nhid
                 , nclass
                 , nlayers = 3
                 , dropout = 0
                 , weight_decay = 5e-4
                 , batchnorm=False,with_bias=True, device='cuda'):
        super(GCN, self).__init__()
        self.with_bn = False
        self.layers = torch.nn.ModuleList()
        # self.layers.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
        self.layers.append(GCNConv(nfeat, nhid, cached=True))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(nhid))
        for _ in range(nlayers - 2):
            self.layers.append(
                GCNConv(nhid, nhid, cached=True))
                # GraphConvolution(nhid, nhid, with_bias=with_bias))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.layers.append(GCNConv(nhid, nclass, cached=True))
        # self.layers.append(GraphConvolution(nhid, nclass, with_bias=with_bias))

        self.dropout = dropout


    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    # def forward(self, x, edge_index: Union[Tensor, SparseTensor]):
    def forward(self, x, adj):
        for i, layers in enumerate(self.layers[:-1]):
            x = layers(x, adj)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i == len(self.layers) - 2:  # 第一层
                x_out = x  # 保留当前层的嵌入表示
        x = self.layers[-1](x, adj)
        return x_out, x.log_softmax(dim=-1)

    def initialize(self):
        """
           Initialize parameters of GCN.
        """
        for layer in self.layers:  # 对每层进行初始化操作
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def predict(model, features=None, adj=None, no_conv=False):
        model.eval()

        if no_conv:
            out = model(features)
        else:
            out = model(features, adj)
        # y_pred = out.exp()

        # aucs = []
        # for mask in masks:
        #     test_auc = metrics.roc_auc_score(data.y[mask].cpu().numpy(), y_pred[mask, 1].detach().cpu().numpy())
        #     aucs.append(test_auc)
        return out

class GraphConvolution(Module):

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # 权重矩阵 128x256
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()  # 重新设置参数

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.T.size(1))  # stdv 是标准差（standard deviation）的倒数，用于设置权重初始化的范围
        self.weight.data.uniform_(-stdv, stdv)  # 让权重的初始值分布在对称区间内，且分布均匀
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)  # 初始化偏差，方式与权重相同

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)  # XxW
        if isinstance(adj, torch_sparse.SparseTensor):
            output = torch_sparse.matmul(adj, support)
        else:
            output = torch.spmm(adj, support)  # AxXxW
        if self.bias is not None:
            return output + self.bias  # AxXxW+b
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

