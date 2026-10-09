import scipy.sparse as sp
import numpy as np
import torch_geometric.transforms as T
import sys
import json
import os
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import InMemoryDataset, Data
import torch
from itertools import repeat
from torch_geometric.data import NeighborSampler
from scipy.sparse import csr_matrix
class DataGraphSAINT:
    '''datasets used in GraphSAINT paper'''

    def __init__(self, dataset, **kwargs):
        dataset_str='data/'+dataset+'/'
        adj_full = sp.load_npz(dataset_str+'adj_full.npz')  # 载入完全邻接矩阵 （169343，169343）
        self.nnodes = adj_full.shape[0]  # 节点数
        if dataset == 'ogbn-arxiv':
            adj_full = adj_full + adj_full.T  # 转为无向图
            adj_full[adj_full > 1] = 1  # 转为无权图

        role = json.load(open(dataset_str+'role.json','r'))  # 训练 验证 测试集索引

        # 训练验证测试集的索引
        idx_train = role['tr']  # 90941
        idx_test = role['te']  # 29799
        idx_val = role['va']  # 48603

        if 'label_rate' in kwargs:
            label_rate = kwargs['label_rate']
            if label_rate < 1:
                idx_train = idx_train[:int(label_rate*len(idx_train))]
        # 对多行多列同时赋值  获取对应的邻接矩阵(稀疏矩阵)
        self.adj_train = adj_full[np.ix_(idx_train, idx_train)]
        self.adj_val = adj_full[np.ix_(idx_val, idx_val)]
        self.adj_test = adj_full[np.ix_(idx_test, idx_test)]

        feat = np.load(dataset_str+'feats.npy')  # （169343，128）
        # ---- normalize feat ----
        feat_train = feat[idx_train]
        scaler = StandardScaler()  # 数据标准化  创建了一个标准化器，用于缩放数据
        scaler.fit(feat_train)  # 根据训练数据 feat_train 计算特征的均值和标准差
        feat = scaler.transform(feat)  # 根据上一步得到的均值和标准差对 feat 进行标准化处理，转换后的数据存储在 feat 中

        self.feat_train = feat[idx_train]  # (90941,128)
        self.feat_val = feat[idx_val]  # (29799,128)
        self.feat_test = feat[idx_test]  # (48603,128)

        class_map = json.load(open(dataset_str + 'class_map.json','r'))  # 169343 dict 每个元素为节点对应类
        labels = self.process_labels(class_map)  # 得到的标签数组

        self.labels_train = labels[idx_train]  # 训练集标签
        self.labels_val = labels[idx_val]  # 验证集标签
        self.labels_test = labels[idx_test]  # 测试集标签

        self.data_full = GraphData(adj_full, feat, labels, idx_train, idx_val, idx_test)   # 全部数据存储  包含邻接矩阵 特征 标签 训练集、验证集、测试集索引
        self.class_dict = None
        self.class_dict2 = None

        self.adj_full = adj_full
        self.feat_full = feat
        self.labels_full = labels
        self.idx_train = np.array(idx_train)
        self.idx_val = np.array(idx_val)
        self.idx_test = np.array(idx_test)
        self.samplers = None

    def process_labels(self, class_map): # 类标签处理
        """
        setup vertex property map for output classests
        """
        num_vertices = self.nnodes  # 节点数
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
            self.nclass = num_classes
            class_arr = np.zeros((num_vertices, num_classes))
            for k,v in class_map.items():
                class_arr[int(k)] = v
        else:
            class_arr = np.zeros(num_vertices, dtype=np.int_)  # 标签数组
            for k, v in class_map.items():
                class_arr[int(k)] = v
            class_arr = class_arr - class_arr.min()
            self.nclass = max(class_arr) + 1
        return class_arr

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None):
        if args.nlayers == 1:
            sizes = [30]
        if args.nlayers == 2:
            if args.dataset in ['reddit', 'flickr']:
                if args.option == 0:
                    sizes = [15, 8]
                if args.option == 1:
                    sizes = [20, 10]
                if args.option == 2:
                    sizes = [25, 10]
            else:
                sizes = [10, 5]

        if self.class_dict2 is None:
            print(sizes)
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx_train = np.array(self.idx_train)
                    idx = idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx

        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                if len(node_idx) == 0:
                    continue

                self.samplers.append(NeighborSampler(adj,
                                    node_idx=node_idx,
                                    sizes=sizes, batch_size=num,
                                    num_workers=8, return_e_id=False,
                                    num_nodes=adj.size(0),
                                    shuffle=True))
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[c].sample(batch)
        return out

    def retrieve_class_sampler_v2(self, dataset, c, adj, transductive, num=256, nlayers=2, option=0):
        if nlayers == 1:
            sizes = [30]
        if nlayers == 2:
            if dataset in ['reddit', 'flickr']:
                if option == 0:
                    sizes = [15, 8]
                if option == 1:
                    sizes = [20, 10]
                if option == 2:
                    sizes = [25, 10]
            else:
                sizes = [10, 5]

        if self.class_dict2 is None:
            print(sizes)
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx_train = np.array(self.idx_train)
                    idx = idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx

        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                if len(node_idx) == 0:
                    continue

                self.samplers.append(NeighborSampler(adj,
                                    node_idx=node_idx,
                                    sizes=sizes, batch_size=num,
                                    num_workers=4, return_e_id=False,
                                    num_nodes=adj.size(0),
                                    shuffle=True))
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[c].sample(batch)
        return out


    def retrieve_class_sampler_v3(self,transductive, num_per_class=64):
        #num = num_per_class*self.nclass
        if self.class_dict2 is None:
            self.class_dict2 = {}
            node_idx = []
            for i in range(self.nclass):
                if transductive:
                    idx_train = np.array(self.idx_train)
                    idx = idx_train[self.labels_train == i]
                else:
                    #tmp = np.arange(len(self.labels_train))
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx
                node_idx.append(np.random.permutation(self.class_dict2[i])[:num_per_class])
            return np.array(node_idx).reshape(-1)
        
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

    def retrieve_class_sampler_train(self,transductive, num_per_class=64):
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

class GraphData:

    def __init__(self, adj, features, labels, idx_train, idx_val, idx_test):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test


class Data2Pyg:

    def __init__(self, data, device='cuda', transform=None, **kwargs):
        self.data_train = Dpr2Pyg(data.data_train, transform=transform)[0].to(device)
        self.data_val = Dpr2Pyg(data.data_val, transform=transform)[0].to(device)
        self.data_test = Dpr2Pyg(data.data_test, transform=transform)[0].to(device)
        self.nclass = data.nclass
        self.nfeat = data.nfeat
        self.class_dict = None

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.data_train.y == i).cpu().numpy()
        idx = np.arange(len(self.data_train.y))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]


class Dpr2Pyg(InMemoryDataset):

    def __init__(self, dpr_data, transform=None, **kwargs):
        root = 'data/' # dummy root; does not mean anything
        self.dpr_data = dpr_data
        super(Dpr2Pyg, self).__init__(root, transform)
        pyg_data = self.process()
        self.data, self.slices = self.collate([pyg_data])
        self.transform = transform

    def process(self):
        dpr_data = self.dpr_data
        edge_index = torch.LongTensor(dpr_data.adj.nonzero())
        # by default, the features in pyg data is dense
        if sp.issparse(dpr_data.features):
            x = torch.FloatTensor(dpr_data.features.todense()).float()
        else:
            x = torch.FloatTensor(dpr_data.features).float()
        y = torch.LongTensor(dpr_data.labels)
        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = None
        data.val_mask = None
        data.test_mask = None
        return data


    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.data.__cat_dim__(key, item)] = slice(slices[idx],
                                                   slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _download(self):
        pass

class DgraphFinData:
    def __init__(self, dataset, **kwargs):

        # dataset_str = 'data/' + dataset + '/skeleton/'
        # zip = sp.load_npz(dataset_str + 'skeleton_gamma.npy')
        # zip = np.load(dataset_str + 'skeleton_gamma.npy',allow_pickle=True).item()
        # self.nnodes = zip['x'].shape[0]  # 节点数
        # edge_index = torch.from_numpy(zip['edge_index']).type(torch.int64)

        file_path = 'data/DGraphFin/dgraphfin.npz'
        dataset = np.load(file_path)
        x = torch.from_numpy(dataset['x'])
        # x = (x - x.mean(0)) / x.std(0)
        y = torch.from_numpy(dataset['y']).to('cuda:1')
        edge_index = torch.from_numpy(dataset['edge_index'].T)
        self.nnodes = x.shape[0]  # 节点数
        # adj_full = edge_index + edge_index.T  # 转为无向图
        # adj_full[adj_full > 1] = 1
        # if dataset == 'ogbn-arxiv':
        #     adj_full = adj_full + adj_full.T  # 转为无向图
        #     adj_full[adj_full > 1] = 1  # 转为无权图
        # x = torch.from_numpy(zip['x'])
        # x = (x - x.mean(0)) / x.std(0)
        # y = torch.from_numpy(zip['y'])
        edge_attr = None
        # data = Data(x=x.float(), edge_index=edge_index, edge_attr=edge_attr, y=y)
        # transform = T.ToSparseTensor()
        # data = transform(data)
        # data.adj_t = data.adj_t.to_symmetric()
        # coo_data = data.adj_t.coo()
        adj_matrix_csr = csr_matrix((torch.ones(dataset['edge_index'].shape[0]).numpy(), (edge_index[0].numpy(), edge_index[1].numpy())), shape=(self.nnodes, self.nnodes))
        adj_matrix_csr = adj_matrix_csr + adj_matrix_csr.T  # 转为无向图
        adj_matrix_csr[adj_matrix_csr > 1] = 1
        # role = json.load(open(dataset_str + 'role.json', 'r'))
        train_mask = torch.from_numpy(dataset['train_mask'])
        valid_mask = torch.from_numpy(dataset['valid_mask'])
        test_mask = torch.from_numpy(dataset['test_mask'])
        idx_train = train_mask
        idx_val = valid_mask
        idx_test = test_mask
        # 训练验证测试集的索引
        # idx_train = zip['train_mask']
        # idx_val = zip['valid_mask']
        # idx_test = zip['test_mask']
        if 'label_rate' in kwargs:
            label_rate = kwargs['label_rate']
            if label_rate < 1:
                idx_train = idx_train[:int(label_rate * len(idx_train))]
        # 对多行多列同时赋值  获取对应的邻接矩阵
        self.adj_train = adj_matrix_csr[np.ix_(idx_train, idx_train)]
        self.adj_val = adj_matrix_csr[np.ix_(idx_val, idx_val)]
        self.adj_test = adj_matrix_csr[np.ix_(idx_test, idx_test)]

        feat = x
        # ---- normalize feat ----
        feat_train = feat[idx_train]
        scaler = StandardScaler()  # 数据标准化  创建了一个标准化器，用于缩放数据
        scaler.fit(feat_train)  # 根据训练数据 feat_train 计算特征的均值和标准差
        feat = scaler.transform(feat)  # 根据上一步得到的均值和标准差对 feat 进行标准化处理，转换后的数据存储在 feat 中

        self.feat_train = feat[idx_train]  # (90941,128)
        self.feat_val = feat[idx_val]  # (29799,128)
        self.feat_test = feat[idx_test]  # (48603,128)

        # class_map = json.load(open(dataset_str + 'class_map.json', 'r'))  # 169343 dict
        # labels = self.process_labels(class_map)  # 得到的标签数组
        labels = y.cpu().numpy()

        self.labels_train = labels[idx_train]  # 训练集标签
        self.labels_val = labels[idx_val]  # 验证集标签
        self.labels_test = labels[idx_test]  # 测试集标签

        self.data_full = GraphData(adj_matrix_csr, feat, labels, idx_train, idx_val,
                                   idx_test)  # 全部数据存储  包含邻接矩阵 特征 标签 训练集、验证集、测试集索引
        self.class_dict = None
        self.class_dict2 = None

        self.adj_full = adj_matrix_csr
        self.feat_full = feat
        self.labels_full = labels
        self.idx_train = np.array(idx_train)
        self.idx_val = np.array(idx_val)
        self.idx_test = np.array(idx_test)
        self.samplers = None
        # self.nclass = len(np.unique(labels))
        self.nclass = 2