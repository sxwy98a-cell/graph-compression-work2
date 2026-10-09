import random
import argparse

from get_data import get_data
# from configs import load_config
from utils import *
from utils_graphsaint import DataGraphSAINT, DgraphFinData
from models.gcn import GCN
from coreset import KCenter, Herding, Random
from tqdm import tqdm
import torch
import deeprobust.graph.utils as utils
import datetime
import os
import sys
import numpy as np
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
# parser.add_argument('--dataset', type=str, default='DGraphFin')
parser.add_argument('--dataset', type=str, default='ogbn-mag')
parser.add_argument('--hidden', type=float, default=256)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--lr_coreset', type=float, default=0.01)
parser.add_argument('--wd_coreset', type=float, default=5e-4)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--nlayers', type=int, default=2, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--save_log', type=str, default='logs')
parser.add_argument('--method', type=str, default='random', choices=['kcenter', 'herding', 'random'])
parser.add_argument('--reduction_rate', type=float, default=0.4)
parser.add_argument('--load_npy', type=str, default='')
parser.add_argument('--opt_type_train', type=str, default='Adam')
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()

device = torch.device(args.device)
log_dir = './' + args.save_log + '/Coreset/{}-reduce_{}-{}-temp'.format(args.dataset, str(args.reduction_rate),
                                                                   datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))  # 日志文件地址和名称
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_format = '%(asctime)s %(message)s'  # 定义了日志输出的格式
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(log_dir, 'coreset.log'))  # 创建一个文件处理器，将日志输出到指定的日志文件中
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info('This is the log_dir: {}'.format(log_dir))

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)
    data_full = data.data_full
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
elif args.dataset in ['DGraphFin']:
    data = DgraphFinData(args.dataset)
    data_full = data.data_full
else:
    data = get_data(args.dataset)
    # data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)

features, adj, labels = data.feat_full, data.adj_full, data.labels_full
adj, features, labels = utils.to_tensor(adj, features, labels, device=device)
adj, features, labels = adj.to(device), features.to(device), labels.to(device)
if utils.is_sparse_tensor(adj):
    adj_norm = utils.normalize_adj_tensor(adj, sparse=True)  # 对邻接矩阵执行对称归一化
else:
    adj_norm = utils.normalize_adj_tensor(adj)

adj = adj_norm.to(device)

labels_test = labels[data.idx_test]

# 获取所有背景节点和目标节点索引
# ogbn_arxiv数据集
idx_target = np.concatenate((data.idx_val, data.idx_test), axis=0)  # 所有目标节点的索引列表
idx_back = data.idx_train

# idx_train = data.idx_train
# idx_val = data.idx_val
# idx_test = data.idx_test
# idx_target = np.concatenate((idx_train, idx_val, idx_test), axis=0)  # 所有目标节点的索引列表
# mask = np.ones(data.nnodes, dtype=bool)
# mask[idx_target] = False  # 将部分索引置为 False
# idx_background = np.arange(data.nnodes)[mask]



# Setup GCN Model
# device = 'cuda'
model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=data.nclass, device=device,
            weight_decay=args.weight_decay)

model = model.to(device)

if args.load_npy=='':
    optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for e in range(args.epochs + 1):
        model.train()
        optimizer_model.zero_grad()
        embed, output = model.forward(features, adj)  # 返回最开始的嵌入表示 (169343,256)
        loss = F.nll_loss(output[data.idx_train], labels[data.idx_train])  # # 负对数似然损失
        acc = utils.accuracy(output[data.idx_train], labels[data.idx_train])

        logging.info('============Train===============')
        logging.info(
            'Epochs={}: Full graph train set results: loss = {:.4f}, accuracy = {:.4f}'.format(e, loss.item(), acc.item()))
        loss.backward()
        optimizer_model.step()
        if e % 10 == 0:  # 每十轮进行一次梯度下降
            model.eval()
            _, output_test = model.forward(features, adj)
            loss_test = F.nll_loss(output_test[data.idx_test], labels_test)
            acc_test = utils.accuracy(output_test[data.idx_test], labels_test)
            logging.info('=========Testing===============')
            logging.info(
                'Epochs={}: Test results: loss = {:.4f}, accuracy = {:.4f}'.format(e, loss_test.item(), acc_test.item()))

        embed_out = embed

    if args.method == 'kcenter':
        agent = KCenter(data, args, device=device)
    if args.method == 'herding':
        agent = Herding(data, args, device=device)
    if args.method == 'random':
        agent = Random(data, args, device=device)
    idx_selected = agent.select(embed_out)  # 所选的节点索引列表
    feat_train = features[idx_selected]  # 选取他们的特征
    adj_train = data.adj_full[np.ix_(idx_selected, idx_selected)]  # 选取所选节点的邻接矩阵
    labels_train = labels[idx_selected]  # 获得所选节点的标签

    s_idx_train = np.concatenate((idx_target, idx_selected), axis=0)
    s_adj_train = data.adj_full[np.ix_(s_idx_train, s_idx_train)]  # 选取所选节点的邻接矩阵
    s_labels_train = labels[s_idx_train]



    if args.save:
        logging.info('Saving...')
        np.save(f'{log_dir}/idx_{args.dataset}_{args.reduction_rate}_{args.method}_{args.seed}.npy', s_idx_train)  # 将所选的节点序号存储到选择文件夹中
    logging.info(args)
    logging.info(log_dir)
else:
    res = []
    #runs = 10
    logging.info('Loading from: {}'.format(args.load_npy))
    idx_selected_train = np.load(f'{args.load_npy}/idx_{args.dataset}_{args.reduction_rate}_{args.method}_{args.seed}.npy')
    feat_train = features[idx_selected_train]
    adj_train = data.adj_full[np.ix_(idx_selected_train, idx_selected_train)]
    labels_train = labels[idx_selected_train]
    if sp.issparse(adj_train):
        adj_train = sparse_mx_to_torch_sparse_tensor(adj_train)
    else:
        adj_train = torch.FloatTensor(adj_train)
    adj_train, feat_train, labels_train = adj_train.to(device), feat_train.to(device), labels_train.to(device)

    if utils.is_sparse_tensor(adj_train):
        adj_train_norm = utils.normalize_adj_tensor(adj_train, sparse=True)
    else:
        adj_train_norm = utils.normalize_adj_tensor(adj_train)

    adj_train = adj_train_norm.to(device)
    if args.opt_type_train=='Adam':
        optimizer_model_coreset = torch.optim.Adam(model.parameters(), lr=args.lr_coreset, weight_decay=args.wd_coreset)
    elif args.opt_type_train=='SGD':
        optimizer_model_coreset = torch.optim.SGD(model.parameters(), lr=args.lr_coreset, momentum=0.9)
    for _ in tqdm(range(args.runs)):
        model.initialize()
        best_test_acc=0
        for e in range(args.epochs + 1):
            model.train()
            optimizer_model_coreset.zero_grad()
            _, output_train = model.forward(feat_train, adj_train)
            loss_train = F.nll_loss(output_train, labels_train)
            acc_train = utils.accuracy(output_train, labels_train)
            logging.info('=========Train coreset===============')
            logging.info('Epochs={}: coreset results: loss = {:.4f}, accuracy = {:.4f}'.format(e, loss_train.item(),
                                                                                               acc_train.item()))

            loss_train.backward()
            optimizer_model_coreset.step()
            if e % 10 == 0:
                model.eval()
                _, output_test = model.forward(features, adj)
                loss_test = F.nll_loss(output_test[idx_test], labels_test)
                acc_test = utils.accuracy(output_test[idx_test], labels_test)
                if acc_test > best_test_acc:
                    best_test_acc = acc_test.item()
                logging.info('=========Test coreset===============')
                logging.info('Epochs={}: Test coreset results: loss = {:.4f}, accuracy = {:.4f}'.format(e, loss_test.item(),
                                                                                                   acc_test.item()))
        res.append(best_test_acc)

    res = np.array(res)
    logging.info(args)
    logging.info(log_dir)
    logging.info('Mean accuracy = {:.4f}, Std = {:.4f}'.format(res.mean(), res.std()))
