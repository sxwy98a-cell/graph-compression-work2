import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from itertools import product
import numpy as np


class PGE(nn.Module):

    def __init__(self, nfeat, nnodes, nhid=256, nlayers=3, device='cuda:1', args=None, data=None, idx_target=None, idx_back=None, adj_syn=None):
        super(PGE, self).__init__()
        if args.dataset in ['ogbn-arxiv', 'arxiv', 'flickr']:
           nhid = 256
        if args.dataset in ['reddit']:
           nhid = 256
           if args.reduction_rate==0.01:
               nhid = 128
           nlayers = 3
           # nhid = 128
        self.idx_target = idx_target
        self.idx_back = idx_back
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(nfeat*2, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhid))
        for i in range(nlayers-2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.layers.append(nn.Linear(nhid, 1))

        self.data = data
        # edge_index = np.array(list(product(range(nnodes), range(nnodes))))
        # row, col = np.meshgrid(range(nnodes), range(nnodes), indexing='ij')
        # edge_index = np.stack([row.ravel(), col.ravel()], axis=1)

        # row = torch.arange(nnodes, device='cuda').repeat(nnodes)
        # col = row.view(nnodes, nnodes).T.flatten()
        # edge_index = torch.stack([row, col], dim=1)
        edge_index = self.data.adj_full

        adj_target = edge_index[idx_target, :][:, idx_target]
        adj_back = edge_index[idx_back, :][:, idx_back]
        self.adj_syn = adj_syn
        self.adj_target = adj_target
        self.adj_back = adj_back
        self.edge_index = edge_index.T
        self.nnodes = nnodes
        self.device = device
        self.reset_parameters()
        self.cnt = 0
        self.args = args
        self.nnodes = nnodes

    # def forward(self, x, edge_idx, inference=False):
    #     if self.args.dataset == 'reddit' and self.args.reduction_rate >= 0.01:
    #         edge_index = self.edge_index
    #         n_part = 5
    #         splits = np.array_split(np.arange(edge_index.shape[1]), n_part)
    #         edge_embed = []
    #         for idx in splits:
    #             tmp_edge_embed = torch.cat([x[edge_index[0][idx]], x[edge_index[1][idx]]], dim=1)
    #             for ix, layer in enumerate(self.layers):
    #                 tmp_edge_embed = layer(tmp_edge_embed)
    #                 if ix != len(self.layers) - 1:
    #                     tmp_edge_embed = self.bns[ix](tmp_edge_embed)
    #                     tmp_edge_embed = F.relu(tmp_edge_embed)
    #             edge_embed.append(tmp_edge_embed)
    #         edge_embed = torch.cat(edge_embed)
    #     else:
    #         edge_idx = edge_idx.T
    #         edge_embed = torch.cat([x[edge_idx[0]], x[edge_idx[1]]], dim=1)  # 1335534
    #         for ix, layer in enumerate(self.layers):
    #             edge_embed = layer(edge_embed)
    #             if ix != len(self.layers) - 1:
    #                 edge_embed = self.bns[ix](edge_embed)
    #                 edge_embed = F.relu(edge_embed)
    #
    #     # indices_test = torch.arange(self.nnodes).repeat(2, 1).to(self.device)  # 对角线的行列索引
    #     # values_test = torch.ones(self.nnodes, device=self.device)
    #     indices = edge_idx
    #     indices = torch.tensor(indices).to(self.device)
    #
    #     values = edge_embed.squeeze().to(self.device)
    #     values = torch.sigmoid(values)  # 仅对边权重应用 sigmoid
    #     adj = torch.sparse_coo_tensor(indices, values, size=(self.nnodes, self.nnodes))
    #
    #     return adj

    # 1264626 1437158
    def forward(self, x, edge_idx, return_logits=False):



        if isinstance(edge_idx, np.ndarray):
            edge_idx = torch.from_numpy(edge_idx).long().to(x.device)
        else:
            edge_idx = edge_idx.to(x.device)
        edge_idx = edge_idx.T
        edge_embed = torch.cat([x[edge_idx[0]], x[edge_idx[1]]], dim=1)

        for ix, layer in enumerate(self.layers):
            edge_embed = layer(edge_embed)
            if ix != len(self.layers) - 1:
                edge_embed = self.bns[ix](edge_embed)
                edge_embed = F.relu(edge_embed)

        logits = edge_embed.squeeze()  # (E,)

        if return_logits:
            return logits  #

        # ===== inference only =====
        values = torch.sigmoid(logits)
        indices = edge_idx.to(self.device)


        adj = torch.sparse_coo_tensor(
            indices, values,
            size=(self.nnodes, self.nnodes)
        )

        return adj


    @torch.no_grad()
    def inference(self, x, edge_idx):
        # self.eval()
        adj_syn = self.forward(x, edge_idx, inference=True)

        return adj_syn

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)