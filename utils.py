import os.path as osp
import random

import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import get_train_val_test
from torch_geometric.utils import train_test_split_edges
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from deeprobust.graph.utils import *
from torch_geometric.data import NeighborSampler
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
import math
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.cuda.device_count()
import logging


def get_dataset(name, normalize_features=True, transform=None, if_dpr=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, name)
    elif name in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    else:
        raise NotImplementedError

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    dpr_data = Pyg2Dpr(dataset)
    if name in ['ogbn-arxiv']:
        # the features are different from the features provided by GraphSAINT
        # normalize features, following graphsaint
        feat, idx_train = dpr_data.features, dpr_data.idx_train
        feat_train = feat[idx_train]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat = scaler.transform(feat)
        dpr_data.features = feat

    return dpr_data


class Pyg2Dpr(Dataset):
    def __init__(self, pyg_data, **kwargs):
        try:
            splits = pyg_data.get_idx_split()
        except:
            pass

        dataset_name = pyg_data.name
        pyg_data = pyg_data[0]
        n = pyg_data.num_nodes

        if dataset_name == 'ogbn-arxiv': # symmetrization
            pyg_data.edge_index = to_undirected(pyg_data.edge_index, pyg_data.num_nodes)

        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),
            (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))

        self.features = pyg_data.x.numpy()
        self.labels = pyg_data.y.numpy()

        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1) # ogb-arxiv needs to reshape

        if hasattr(pyg_data, 'train_mask'):
            # for fixed split
            self.idx_train = mask_to_index(pyg_data.train_mask, n)
            self.idx_val = mask_to_index(pyg_data.val_mask, n)
            self.idx_test = mask_to_index(pyg_data.test_mask, n)
            self.name = 'Pyg2Dpr'
        else:
            try:
                # for ogb
                self.idx_train = splits['train']
                self.idx_val = splits['valid']
                self.idx_test = splits['test']
                self.name = 'Pyg2Dpr'
            except:
                # for other datasets
                self.idx_train, self.idx_val, self.idx_test = get_train_val_test(
                        nnodes=n, val_size=0.1, test_size=0.8, stratify=self.labels)


def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask



class Transd2Ind:
    # transductive setting to inductive setting

    def __init__(self, dpr_data, keep_ratio=1.0):
        idx_train, idx_val, idx_test = dpr_data.idx_train, dpr_data.idx_val, dpr_data.idx_test
        adj, features, labels = dpr_data.adj, dpr_data.features, dpr_data.labels
        self.nclass = labels.max()+1  # 类别数
        self.adj_full, self.feat_full, self.labels_full = adj, features, labels
        self.idx_train = np.array(idx_train)
        self.idx_val = np.array(idx_val)
        self.idx_test = np.array(idx_test)

        if keep_ratio < 1:
            idx_train, _ = train_test_split(idx_train,
                                            random_state=None,
                                            train_size=keep_ratio,
                                            test_size=1-keep_ratio,
                                            stratify=labels[idx_train])

        self.adj_train = adj[np.ix_(idx_train, idx_train)]
        self.adj_val = adj[np.ix_(idx_val, idx_val)]
        self.adj_test = adj[np.ix_(idx_test, idx_test)]
        logging.info('size of adj_train: {}'.format(self.adj_train.shape))
        logging.info('#edges in adj_train: {}'.format(self.adj_train.sum()))

        self.labels_train = labels[idx_train]
        self.labels_val = labels[idx_val]
        self.labels_test = labels[idx_test]

        self.feat_train = features[idx_train]
        self.feat_val = features[idx_val]
        self.feat_test = features[idx_test]

        self.class_dict = None
        self.samplers = None
        self.class_dict2 = None

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx

        if args.nlayers == 1:
            sizes = [15]
        if args.nlayers == 2:
            sizes = [10, 5]
            # sizes = [-1, -1]
        if args.nlayers == 3:
            sizes = [15, 10, 5]
        if args.nlayers == 4:
            sizes = [15, 10, 5, 5]
        if args.nlayers == 5:
            sizes = [15, 10, 5, 5, 5]


        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                self.samplers.append(NeighborSampler(adj,
                                    node_idx=node_idx,
                                    sizes=sizes, batch_size=num,
                                    num_workers=4, return_e_id=False,
                                    num_nodes=adj.size(0),
                                    shuffle=True))
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[c].sample(batch)
        return out

    def retrieve_class_multi_sampler(self, c, adj, transductive, num=256, args=None):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx


        if self.samplers is None:
            self.samplers = []
            for l in range(2):
                layer_samplers = []
                sizes = [15] if l == 0 else [10, 5]
                for i in range(self.nclass):
                    node_idx = torch.LongTensor(self.class_dict2[i])
                    layer_samplers.append(NeighborSampler(adj,
                                        node_idx=node_idx,
                                        sizes=sizes, batch_size=num,
                                        num_workers=12, return_e_id=False,
                                        num_nodes=adj.size(0),
                                        shuffle=True))
                self.samplers.append(layer_samplers)
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[args.nlayers-1][c].sample(batch)
        return out

    def retrieve_class_sampler_val(self,transductive, num_per_class=64):
        #num = num_per_class*self.nclass
        if self.class_dict2 is None:
            self.class_dict2 = {}
            node_idx = []
            for i in range(self.nclass):
                if transductive:
                    idx_val = np.array(self.idx_val)
                    idx = idx_val[self.labels_val == i]
                else:
                    idx = np.arange(len(self.labels_val))[self.labels_val==i]
                self.class_dict2[i] = idx
                #node_idx.append(np.random.permutation(self.class_dict2[i])[:num_per_class])
                node_idx += np.random.permutation(self.class_dict2[i])[:num_per_class].tolist()
            self.class_dict2 = None
            return np.array(node_idx).reshape(-1)

def match_loss(gw_syn, gw_real, args, device):
    dis = torch.tensor(0.0).to(device)

    if args.dis_metric == 'ours':

        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis

def distance_wb(gwr, gws):
    shape = gwr.shape

    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis



def calc_f1(y_true, y_pred,is_sigmoid):
    if not is_sigmoid:
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def evaluate(output, labels, args):
    data_graphsaint = ['yelp', 'ppi', 'ppi-large', 'flickr', 'reddit', 'amazon']
    if args.dataset in data_graphsaint:
        labels = labels.cpu().numpy()
        output = output.cpu().numpy()
        if len(labels.shape) > 1:
            micro, macro = calc_f1(labels, output, is_sigmoid=True)
        else:
            micro, macro = calc_f1(labels, output, is_sigmoid=False)
        logging.info("Test set results:", "F1-micro= {:.4f}".format(micro),
                "F1-macro= {:.4f}".format(macro))
    else:
        loss_test = F.nll_loss(output, labels)
        acc_test = accuracy(output, labels)
        logging.info("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
    return


from torchvision import datasets, transforms
def get_mnist(data_path):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no        augmentation
    dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    class_names = [str(c) for c in range(num_classes)]

    labels = []
    feat = []
    for x, y in dst_train:
        feat.append(x.view(1, -1))
        labels.append(y)
    feat = torch.cat(feat, axis=0).numpy()
    from utils_graphsaint import GraphData
    adj = sp.eye(len(feat))
    idx = np.arange(len(feat))
    dpr_data = GraphData(adj-adj, feat, labels, idx, idx, idx)
    from deeprobust.graph.data import Dpr2Pyg
    return Dpr2Pyg(dpr_data)

def regularization(adj, x, eig_real=None):
    # fLf
    loss = 0
    # loss += torch.norm(adj, p=1)
    loss += feature_smoothing(adj, x)
    return loss

def maxdegree(adj):
    n = adj.shape[0]
    return F.relu(max(adj.sum(1))/n - 0.5)

def sparsity2(adj):
    n = adj.shape[0]
    loss_degree = - torch.log(adj.sum(1)).sum() / n
    loss_fro = torch.norm(adj) / n
    return 0 * loss_degree + loss_fro

def sparsity(adj):
    n = adj.shape[0]
    thresh = n * n * 0.01
    return F.relu(adj.sum()-thresh)
    # return F.relu(adj.sum()-thresh) / n**2

def feature_smoothing(adj, X):
    adj = (adj.t() + adj)/2
    rowsum = adj.sum(1)
    r_inv = rowsum.flatten()
    D = torch.diag(r_inv)
    L = D - adj

    r_inv = r_inv  + 1e-8
    r_inv = r_inv.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    # L = r_mat_inv @ L
    L = r_mat_inv @ L @ r_mat_inv

    XLXT = torch.matmul(torch.matmul(X.t(), L), X)
    loss_smooth_feat = torch.trace(XLXT)
    # loss_smooth_feat = loss_smooth_feat / (adj.shape[0]**2)
    return loss_smooth_feat

def row_normalize_tensor(mx):
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1).flatten()
    # r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    return mx

def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = [model, 'GAT', 'MLP', 'APPNP', 'GraphSage', 'Cheby','GCN']
    elif eval_mode == 'S': # itself
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]  # 去掉BN
    else:
        model_eval_pool = [model_eval]
        
    return model_eval_pool








device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# neighborhood-based difficulty measurer
def neighborhood_difficulty_measurer(data, adj, label):
    # edge_value = adj[np.ix_(data.idx_target,data.idx_target)].coalesce().values()  # 提取每条边的值

    # dense_adj = adj.to_dense()
    # dense_sub_adj = dense_adj[np.ix_(data.idx_target,data.idx_target)] # 将稀疏矩阵中的重复项合并并规范化，将矩阵转化为稀疏矩阵
    # sparse_sub_adj = dense_sub_adj.to_sparse_coo()  # 确保 PyTorch 版本 >= 1.10
    # edge_index = sparse_sub_adj.coalesce().indices()

    edge_index = adj.coalesce().indices()
    neighbor_label, _ = add_self_loops(edge_index)  # 添加自环 #[[1, 1, 1, 1],[2, 3, 4, 5]]

    neighbor_label[1] = label[neighbor_label[1]]   # 将所有边 映射到类上 [[1, 1, 1, 1],[40, 20, 19, 21]]

    neighbor_label = torch.transpose(neighbor_label, 0, 1)  # # 进行 0 维和 1 维的维度转换 [[1, 40], [1, 20], [1, 19], [1, 21]]

    index, count = torch.unique(neighbor_label, sorted=True, return_counts=True, dim=0)  # 消除重复元素 并用count对重复元素计数

    neighbor_class = torch.sparse_coo_tensor(index.T, count)  # 转化为一个稀疏矩阵 每个元素代表一个点与对应类连边的个数
    neighbor_class = neighbor_class.to_dense().float()  # 转化为普通邻接矩阵

    neighbor_class = neighbor_class[data.idx_train]  # 找到训练节点集
    neighbor_class = F.normalize(neighbor_class, 1.0, 1)  # 每一行执行一下归一化
    neighbor_entropy = -1 * neighbor_class * torch.log(neighbor_class + torch.exp(torch.tensor(-20)))  # 计算熵 防止log里面是0出现异常
    local_difficulty = neighbor_entropy.sum(1)  # 求和得到难度

    print('done')
    return local_difficulty.to(device)


def neighborhood_difficulty_measurer_in(data, adj, label):
    edge_index = adj.coalesce().indices()
    edge_value = adj.coalesce().values()
    
    neighbor_label, _ = add_self_loops(edge_index)  # 添加自环
 
    neighbor_label[1] = label[neighbor_label[1]]   # 将每一条边的第二个节点转化为对应的类

    neighbor_label = torch.transpose(neighbor_label, 0, 1)  # # 进行 0 维和 1 维的维度转换
  
    index, count = torch.unique(neighbor_label, sorted=True, return_counts=True, dim=0)  # 去掉重复元素 每个元素记录对应个数
    

    neighbor_class = torch.sparse_coo_tensor(index.T, count) # 转化为一个稀疏矩阵 每个元素代表一个点与对应类连边的个数
    neighbor_class = neighbor_class.to_dense().float()  # 转化为普通邻接矩阵
 
    neighbor_class = F.normalize(neighbor_class, 1.0, 1)
    neighbor_entropy = -1 * neighbor_class * torch.log(neighbor_class + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
    local_difficulty = neighbor_entropy.sum(1)

    print('done')
    return local_difficulty.to(device)



def difficulty_measurer(data, adj, label):
    local_difficulty = neighborhood_difficulty_measurer(data, adj, label)  # 基于邻居节点的难度测量
    # global_difficulty = feature_difficulty_measurer(data, label, embedding)
    node_difficulty = local_difficulty 
    return node_difficulty


def sort_training_nodes(data, adj, label):  # 训练节点进行排序
    node_difficulty = difficulty_measurer(data, adj, label)  # 进行节点难度测量 得到每个训练节点一个难度得分
    _, indices = torch.sort(node_difficulty)  # 难度进行排序 得到难度得分排序的序列
    indices = indices.cpu().numpy()

    sorted_trainset = data.idx_train[indices]  # 通过排序序列对训练集进行排序
    return sorted_trainset


def difficulty_measurer_in(data, adj, label):
    local_difficulty = neighborhood_difficulty_measurer_in(data, adj, label)
    # global_difficulty = feature_difficulty_measurer(data, label, embedding)
    node_difficulty = local_difficulty 
    return node_difficulty


def sort_training_nodes_in(data, adj, label):
    node_difficulty = difficulty_measurer_in(data, adj, label)
    _, indices = torch.sort(node_difficulty)
    indices = indices.cpu().numpy()

    return indices



def training_scheduler(lam, t, T, scheduler='geom'):  # 训练调度器的设置
    if scheduler == 'linear':
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == 'root':
        return min(1, math.sqrt(lam ** 2 + (1 - lam ** 2) * t / T))
    elif scheduler == 'geom':
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))
    




# 负样本选择
# def sample_neg_edges_from_target(
#     idx1,
#     idx2,
#     edge_set,
#     num_neg,
#     device
# ):
#     # neg_u = []
#     # neg_v = []
#     #
#     # c_idx_list1 = idx1.tolist()
#     # c_idx_list2 = idx2.tolist()
#     #
#     # while len(neg_u) < num_neg:
#     #     u = random.choice(c_idx_list1)
#     #     v = random.choice(c_idx_list2)
#     #
#     #     if u == v:
#     #         continue
#     #     if ((u, v) in edge_set)|((v, u) in edge_set):
#     #         continue
#     #
#     #     neg_u.append(u)
#     #     neg_v.append(v)
#     neg_u = torch.randint(0, int(idx1.max()), (num_neg,), device=device)
#     neg_v = torch.randint(0, int(idx2.max()), (num_neg,), device=device)
#     edge_index_neg = torch.stack([neg_u, neg_v], dim=0)
#
#     # return torch.tensor([neg_u, neg_v], device=device)
#     return edge_index_neg
def sample_neg_edges_from_target(
    src_candidates,
    dst_candidates,
    pos_edge_index,
    num_samples,
    device,
    oversample_ratio=1.3
):
    """
    src_candidates, dst_candidates: 1D Tensor / list
    pos_edge_index: [2, E]
    num_samples: int
    """

    # ---------- 1. 正边 hash ----------
    pos_set = set(
        (min(u.item(), v.item()), max(u.item(), v.item()))
        for u, v in pos_edge_index
    )

    src = torch.as_tensor(src_candidates, device=device)
    dst = torch.as_tensor(dst_candidates, device=device)

    # ---------- 2. 一次性过采样 ----------
    M = int(num_samples * oversample_ratio)

    u = src[torch.randint(0, src.size(0), (M,), device=device)]
    v = dst[torch.randint(0, dst.size(0), (M,), device=device)]

    cand = torch.stack([u, v], dim=1).cpu().tolist()

    # ---------- 3. 去重 + 过滤正边 ----------
    neg_set = set()
    for e in cand:
        u0, v0 = e[0], e[1]

        if u0 == v0:
            continue  # ← 删除自环

        e = (u0, v0)
        e = (min(e[0], e[1]), max(e[0], e[1]))

        if e not in pos_set and e not in neg_set:
            neg_set.add(e)

        if len(neg_set) >= num_samples:
            break

    # ---------- 4. 不足则补采（极少发生） ----------
    while len(neg_set) < num_samples:
        u = src[torch.randint(0, src.size(0), (1,), device=device)].item()
        v = dst[torch.randint(0, dst.size(0), (1,), device=device)].item()
        e = (u, v)
        e = (min(e[0], e[1]), max(e[0], e[1]))
        et = (max(e[0], e[1]), min(e[0], e[1]))
        if e not in pos_set and e not in neg_set and  et not in pos_set and et not in neg_set:
            neg_set.add(e)

    # ---------- 5. 转回 Tensor ----------
    neg_edges = torch.tensor(
        list(neg_set), device=device
    ).t()

    return neg_edges



def pge_predict_chunk(pge, x, edge_index, chunk_size=50000):
    pge.eval()
    outs = []

    with torch.no_grad():
        E = edge_index.size(0)
        for i in range(0, E, chunk_size):
            sub_edge = edge_index[i:i + chunk_size]
            out = pge.forward(x, sub_edge)
            outs.append(out.cpu())  # 👉 立刻搬到 CPU

    return torch.cat(outs, dim=0)




def aggregate_features_without_network(feat_init, adj_init, add_self_loop=True):
    adj = adj_init.coalesce()
    N = adj.size(0)
    device = feat_init.device


    if add_self_loop:
        idx = torch.arange(N, device=device)
        self_loop = torch.sparse_coo_tensor(
        torch.stack([idx, idx]),
        torch.ones(N, device=device),(N, N))
    adj = (adj + self_loop).coalesce()


    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0


    row, col = adj.indices()
    val = adj.values()
    val = deg_inv_sqrt[row] * val * deg_inv_sqrt[col]


    adj_norm = torch.sparse_coo_tensor(adj.indices(), val, adj.size(), device=adj.device).coalesce()


    feat_agg = torch.sparse.mm(adj_norm, feat_init)
    return feat_agg




def threshold_and_binarize_sparse_adj(
    fin_adj: torch.Tensor,
    threshold: float,
    device=None
):
    """
    对稀疏邻接矩阵做：
    1) coalesce 合并重复边
    2) 小于 threshold 的边删除
    3) 保留边权重设为 1

    参数
    ----
    fin_adj : torch.sparse_coo_tensor, shape [N, N]
    threshold : float
    device : torch.device or None

    返回
    ----
    fin_adj_bin : torch.sparse_coo_tensor
    """

    assert fin_adj.is_sparse, "fin_adj 必须是稀疏张量 (sparse COO)"

    # 1. 合并重复边（不改变图结构语义）
    fin_adj = fin_adj.coalesce()

    indices = fin_adj.indices()   # [2, nnz]
    values = fin_adj.values()     # [nnz]

    # 2. 阈值筛选（只在 values 上做）
    mask = values >= threshold

    if mask.sum() == 0:
        raise RuntimeError("阈值过大，所有边都被删掉了")

    new_indices = indices[:, mask]

    # 3. 二值化：保留的边全部设为 1
    new_values = torch.ones(
        new_indices.shape[1],
        device=values.device if device is None else device
    )

    # 4. 构造新的稀疏邻接矩阵
    fin_adj_bin = torch.sparse_coo_tensor(
        new_indices,
        new_values,
        fin_adj.size(),
        device=fin_adj.device if device is None else device
    ).coalesce()

    return fin_adj_bin





    

