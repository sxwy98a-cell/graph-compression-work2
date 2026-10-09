
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import OGB_MAG
import torch
import scipy.sparse as sp
class GraphData:
    def __init__(self, feat_full, adj_full, labels_full, idx_target, idx_back, idx_train, idx_val, idx_test, nclass, **kwargs):
        self.feat_full = feat_full    # 节点特征
        self.adj_full = adj_full      # 邻接矩阵
        self.labels_full = labels_full  # 节点标签
        self.idx_target = idx_target
        self.idx_back = idx_back
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.nclass = nclass
        # 可选扩展其他属性（如掩码、边特征等）
        for key, value in kwargs.items():
            setattr(self, key, value)
def get_data(my_data):
    if my_data == 'IMDB':
        adj_matrix = sp.load_npz("/root/siton-data-zhengwenpingData/data/2023fyf/GEOM/data/IMDB/adjM.npz")
        feat_0 = sp.load_npz("/root/siton-data-zhengwenpingData/data/2023fyf/GEOM/data/IMDB/features_0.npz")
        feat_1 = sp.load_npz("/root/siton-data-zhengwenpingData/data/2023fyf/GEOM/data/IMDB/features_1.npz")
        feat_2 = sp.load_npz("/root/siton-data-zhengwenpingData/data/2023fyf/GEOM/data/IMDB/features_2.npz")
        idx = np.load("/root/siton-data-zhengwenpingData/data/2023fyf/GEOM/data/IMDB/train_val_test_idx.npz")
        feat_full = sp.vstack([feat_0, feat_1, feat_2])
        feat_full = feat_full.toarray()
        adj_full = adj_matrix
        labels = np.load("/root/siton-data-zhengwenpingData/data/2023fyf/GEOM/data/IMDB/labels.npy")


        # 计算 labels 中最大值 + 1
        new_value = labels.max() + 1
        # 生成补充部分
        pad_size = feat_full.shape[0] - labels.shape[0]
        padding = np.full((pad_size,), new_value, dtype=labels.dtype)
        labels_full = np.concatenate([labels, padding])

        idx_train = idx['train_idx']
        idx_val = idx['val_idx']
        idx_test = idx['test_idx']
        idx_target = np.concatenate([idx_train, idx_val, idx_test])
        idx_full = np.arange(feat_full.shape[0])
        idx_back = np.setdiff1d(idx_full, idx_target)
        nclass = np.unique(labels).shape[0]

        data = GraphData(
            feat_full=feat_full,
            adj_full=adj_full,
            labels_full=labels_full,
            idx_target=idx_target,
            idx_back=idx_back,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            nclass=nclass
        )
        return data
    elif my_data == 'DBLP':
        adj_matrix = sp.load_npz("/root/siton-data-zhengwenpingData/data/2023fyf/GEOM/data/DBLP/adjM.npz")
        feat_0 = sp.load_npz("/root/siton-data-zhengwenpingData/data/2023fyf/GEOM/data/DBLP/features_0.npz")  # 4057 334(目标节点)
        feat_1 = sp.load_npz("/root/siton-data-zhengwenpingData/data/2023fyf/GEOM/data/DBLP/features_1.npz")  # 14328 4231
        feat_2 = np.load("/root/siton-data-zhengwenpingData/data/2023fyf/GEOM/data/DBLP/features_2.npy")  # 7723 50
        feat_2 = csr_matrix(feat_2)
        idx = np.load("/root/siton-data-zhengwenpingData/data/2023fyf/GEOM/data/DBLP/train_val_test_idx.npz")
        feat_full = sp.vstack([feat_0, feat_1, feat_2])
        feat_full = feat_full.toarray()
        adj_full = adj_matrix
        labels = np.load("/root/siton-data-zhengwenpingData/data/2023fyf/GEOM/data/DBLP/labels.npy")

        # 计算 labels 中最大值 + 1
        new_value = labels.max() + 1
        # 生成补充部分
        pad_size = feat_full.shape[0] - labels.shape[0]
        padding = np.full((pad_size,), new_value, dtype=labels.dtype)
        labels_full = np.concatenate([labels, padding])

        idx_train = idx['train_idx']
        idx_val = idx['val_idx']
        idx_test = idx['test_idx']
        idx_target = np.concatenate([idx_train, idx_val, idx_test])
        idx_full = np.arange(feat_full.shape[0])
        idx_back = np.setdiff1d(idx_full, idx_target)
        nclass = np.unique(labels).shape[0]

        data = GraphData(
            feat_full=feat_full,
            adj_full=adj_full,
            labels_full=labels_full,
            idx_target=idx_target,
            idx_back=idx_back,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            nclass=nclass
        )
        return data
    elif my_data == 'ogbn-mag':
        file = '/root/siton-data-zhengwenpingData/data/2023fyf/GEOM/data'
        dataset = OGB_MAG(root=file, preprocess='metapath2vec')
        graph = dataset[0]
        print(graph)
        train_mask = graph['paper']['train_mask']
        train_mask = torch.where(train_mask)[0]
        val_mask = graph['paper']['val_mask']
        val_mask = torch.where(val_mask)[0]
        test_mask = graph['paper']['test_mask']
        test_mask = torch.where(test_mask)[0]
        feat_paper = graph['paper']['x']  # [736389, 128]
        feat_author = graph['author']['x']  # [1134649, 128]
        feat_institution = graph['institution']['x']  # [8740, 128]
        feat_field = graph['field_of_study']['x']  # [59965, 128]
        edge_idx1 = graph['author', 'affiliated_with', 'institution']['edge_index']  # [2, 1043998]  1134648 8739
        edge_idx1[0, :] += 736389
        edge_idx1[1, :] += 1871038
        edge_idx2 = graph['author', 'writes', 'paper']['edge_index']  # [2, 7145660]  1134648 736388
        edge_idx2[0, :] += 736389
        edge_idx3 = graph['paper', 'cites', 'paper']['edge_index']  # [2, 5416271]  736388  736388
        edge_idx4 = graph['paper', 'has_topic', 'field_of_study']['edge_index']  # [2, 7505078] 736388 59964
        edge_idx4[1, :] += 1879778

        edge_idx1 = np.array(edge_idx1)
        edge_idx2 = np.array(edge_idx2)
        edge_idx3 = np.array(edge_idx3)
        edge_idx4 = np.array(edge_idx4)
        edge_idx = np.concatenate([edge_idx1, edge_idx2, edge_idx3, edge_idx4], axis=1)
        num_nodes = max(edge_idx[0].max(), edge_idx[1].max()) + 1
        adj_matrix = csr_matrix((np.ones_like(edge_idx[0]), (edge_idx[0], edge_idx[1])), shape=(num_nodes, num_nodes))

        labels = graph['paper'].y
        new_value = labels.max() + 1
        # 生成补充部分
        pad_size = num_nodes - labels.shape[0]
        padding = np.full((pad_size,), new_value, dtype=labels.numpy().dtype)
        labels_full = np.concatenate([labels, padding])

        nclass = np.unique(labels).shape[0]
        idx_target = np.concatenate([train_mask, val_mask, test_mask])
        feat_full = np.concatenate([feat_paper, feat_author, feat_institution, feat_field])
        idx_full = np.arange(feat_paper.shape[0]+feat_author.shape[0]+feat_institution.shape[0]+feat_field.shape[0])
        idx_back = np.setdiff1d(idx_full, idx_target)
        my_data = GraphData(
            feat_full=feat_full,
            adj_full=adj_matrix,
            labels_full=labels_full,
            idx_target=idx_target,
            idx_back=idx_back,
            idx_train=train_mask,
            idx_val=val_mask,
            idx_test=test_mask,
            nclass=nclass
        )

        return my_data
    elif my_data == 'elliptic':
        df_classes = pd.read_csv(
            '/root/siton-data-zhengwenpingData/data/2023fyf/GEOM/data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')  # 节点标签
        df_features = pd.read_csv(
            '/root/siton-data-zhengwenpingData/data/2023fyf/GEOM/data/elliptic_bitcoin_dataset/elliptic_txs_features.csv',
            header=None)  # 特征
        df_edgelist = pd.read_csv(
            '/root/siton-data-zhengwenpingData/data/2023fyf/GEOM/data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')  # 边
        df_classes = df_classes.to_numpy()
        df_classes = df_classes.T
        labels = np.where(df_classes[1] == "unknown", 3, df_classes[1])
        labels = labels.astype(int)  # array([  4545,  42019, 157205])
        features = df_features.to_numpy()
        features = features[:,1:]
        edge_index = df_edgelist.to_numpy().T
        node_mapping = {node_id: idx for idx, node_id in enumerate(df_classes[0])}
        idx_full = np.arange(features.shape[0])
        idx_target = np.where((labels == 1) | (labels == 2))[0]

        edge_idx = np.vectorize(node_mapping.get)(edge_index)
        num_nodes = features.shape[0]
        adj_matrix = csr_matrix((np.ones_like(edge_idx[0]), (edge_idx[0], edge_idx[1])), shape=(num_nodes, num_nodes))

        train_mask, test_mask = train_test_split(np.arange(features[idx_target].shape[0]), test_size=0.2,
                                                 random_state=15, stratify=labels[idx_target])
        idx_train = idx_target[train_mask]
        idx_test = idx_target[test_mask]
        idx_back = np.setdiff1d(idx_full, idx_target)
        nclass = np.unique(labels).shape[0]
        idx_val = 0
        # labels = np.where(labels == 1, 2, np.where(labels == 2, 1, labels))
        labels = labels-1
        data = GraphData(
            feat_full=features,
            adj_full=adj_matrix,
            labels_full=labels,
            idx_target=idx_target,
            idx_back=idx_back,
            idx_train=idx_train,
            idx_val= idx_val,
            idx_test=idx_test,
            nclass=2
        )
        return data