import gc
from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.sparse import coo_matrix, csr_matrix

from models.m_sage import GraphSage
from models.pge import PGE
from utils import get_eval_pool
import deeprobust.graph.utils as utils
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from models.m_gcn import GCN
from models.sgc import SGC
from models.reparam_module import ReparamModule
import logging
import os
import random
import copy
import scipy
from utils_graphsaint import DataGraphSAINT, DgraphFinData
from utils import *
import wandb


class MetaGtt:

    def __init__(self, data, idx_background, idx_target, args, device='cuda:1', **kwargs):
        self.data = data
        self.args = args
        self.device = device
        self.idx_background = idx_background
        self.idx_target = idx_target
        self.adj_syn = None
        # n = int(s_idx_train * args.reduction_rate)
        n = int(len(self.idx_background) * args.reduction_rate + self.idx_target.shape[0])
        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))
        # self.pge = PGE(nfeat=d, nnodes=n, device=device, args=args).to(device)


        if args.optimizer_con == 'Adam':
            self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)  # 使用Adam优化器对特征进行更新
        elif args.optimizer_con == 'SGD':
            self.optimizer_feat = torch.optim.SGD([self.feat_syn], lr=args.lr_feat, momentum=0.5)
        logging.info('adj_syn: {}, feat_syn: {}'.format((n, n), self.feat_syn.shape))

    def expert_load(self):
        args = self.args
        expert_dir = args.buffer_path  # 获取教师训练路径
        logging.info("Expert Dir: {}".format(expert_dir))

        if args.load_all:
            buffer = []
            n = 0
            while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
                buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(expert_dir))

        else:
            expert_files = []
            n = 0
            while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
                expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))  # 将所有的路径添加
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(expert_dir))
            file_idx = 0
            expert_idx = 0
            random.shuffle(expert_files)
            if args.max_files is not None:
                expert_files = expert_files[:args.max_files]
            # print("loading file {}".format(expert_files[file_idx]))
            buffer = torch.load(expert_files[file_idx])  # 10*100（100为一次网络参数）
            if args.max_experts is not None:
                buffer = buffer[:args.max_experts]
            random.shuffle(buffer)  # 将一个列表中的元素打乱顺序
            self.buffer = buffer

        return file_idx, expert_idx, expert_files

    def synset_save(self):
        args = self.args

        with torch.no_grad():
            feat_save = self.feat_syn
            eval_labs = self.labels_syn

            # 添加动态邻接矩阵部分
        feat_syn_eval, label_syn_eval = copy.deepcopy(feat_save.detach()), copy.deepcopy(
            eval_labs.detach())  # avoid any unaware modification

        # size = feat_syn_eval.shape[0]
        # indices = torch.arange(size).repeat(2, 1).to(self.device)  # 对角线的行列索引
        # values = torch.ones(size, device=self.device)  # 对角线的值为 1
        # adj_syn_eval = torch.sparse_coo_tensor(indices, values, size=(size, size))
        adj_syn_eval = self.adj_syn

        return feat_syn_eval, adj_syn_eval, label_syn_eval

    def eval_synset(self, args):

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        device = torch.device(args.device)
        # logging.info('start!')
        if args.dataset in ['cora', 'citeseer']:
            args.epsilon = 0.05
        else:
            args.epsilon = 0.01

        data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
        if args.dataset in data_graphsaint:
            data = DataGraphSAINT(args.dataset)
            data_full = data.data_full
        elif args.dataset == 'DGraphFin':
            data = DgraphFinData(args.dataset)
        else:
            data_full = get_dataset(args.dataset)
            data = Transd2Ind(data_full)

        res_val = []
        res_test = []
        nlayer = 3
        for i in range(args.nruns):
            best_acc_val, best_acc_test = self.test(args, data, device, model_type=args.test_model_type,
                                                    nruns=i)  # 更新子图
            # _,_ = self.test_lr(args, data, device, model_type=args.test_model_type, nruns=i)

            res_val.append(best_acc_val)
            res_test.append(best_acc_test)
        res_val = np.array(res_val)
        res_test = np.array(res_test)
        logging.info('Model:{}, Layer: {}'.format(args.test_model_type, nlayer))
        logging.info('TEST: Valid Graph BSET Accuracy: {:.6f}, STD: {:.6f}'.format(np.max(res_val), res_val.std()))
        logging.info('TEST: TEST Graph BSET Accuracy: {:.6f}, STD: {:.6f}'.format(np.max(res_test), res_test.std()))


        return res_val, res_test

    def SoftCrossEntropy(self, inputs, target, reduction='average'):
        input_log_likelihood = - inputs
        target_log_likelihood = F.softmax(target, dim=1)
        batch = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
        return loss

    def test(self, args, data, device, model_type, nruns):
        features, adj, labels = data.feat_full, data.adj_full, data.labels_full
        (feat_init, adj_init, labels_init, idx_select_train, idx_target_tra, idx_target_val, idx_target_test,
         idx_target, idx_back, c_idx_target, c_idx_back) = self.get_coreset_init(features, adj, labels)

        feat_init, adj_init, labels_init = utils.to_tensor(feat_init, adj_init, labels_init, device=self.device)
        features_tensor, adj_tensor, labels_tensor = utils.to_tensor(features, adj, labels, device=self.device)

        size = self.feat_syn.shape[0]
        indices = torch.arange(size).repeat(2, 1).to(self.device)  # 对角线的行列索引
        values = torch.ones(size, device=self.device)  # 对角线的值为 1
        adj_syn_eval = torch.sparse_coo_tensor(indices, values, size=(size, size))
        if args.whole_data != 1:
            feat_syn, adj_syn, labels_syn = self.synset_save()

            # size = feat_syn.shape[0]
            # indices = torch.arange(size).repeat(2, 1).to(self.device)  # 对角线的行列索引
            # values = torch.ones(size, device=self.device)  # 对角线的值为 1
            # adj_syn = torch.sparse_coo_tensor(indices, values, size=(size, size)).to(device)
            adj_syn = self.adj_syn

            # adj_syn = torch.eye(feat_syn.shape[0]).to(device)
            if type(adj_syn) is not torch.Tensor:
                feat_syn, adj_syn, labels_syn = utils.to_tensor(feat_syn, adj_syn, labels_syn, device=device)
            else:
                feat_syn, adj_syn, labels_syn = feat_syn.to(device), adj_syn.to(device), labels_syn.to(device)
            if model_type == 'MLP':
                adj_syn = adj_syn - adj_syn
                model_class = GCN
            else:
                model_class = eval(model_type)

            # if utils.is_sparse_tensor(adj_syn):
            #     adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=True)
            # else:
            #     adj_syn_norm = utils.normalize_adj_tensor(adj_syn)
            # adj_syn = adj_syn_norm
            weight_decay = args.test_wd
            lr = args.test_lr_model

        else:
            logging.info('THIS IS THE ORIGINAL WHOLE DATA...')
            features, adj, labels = data.feat_full, data.adj_full, data.labels_full
            features, adj, labels = utils.to_tensor(features, adj, labels, device=device)
            feat_syn, labels_syn = features, labels
            if model_type == 'MLP':
                adj = adj - adj
                model_class = GCN
            else:
                model_class = eval(model_type)
            # if utils.is_sparse_tensor(adj):
            #     adj_syn_norm = utils.normalize_adj_tensor(adj, sparse=True)
            # else:
            #     adj_syn_norm = utils.normalize_adj_tensor(adj)
            #
            # adj_syn = adj_syn_norm
            adj_syn = self.adj_syn
            weight_decay = 5e-4
            lr = 0.01

        # dropout = 0.5 if args.dataset in ['reddit'] else args.test_dropout
        dropout = args.test_dropout
        model_class = GCN
        model = model_class(nfeat=feat_syn.shape[1], nhid=args.test_hidden, dropout=dropout, nlayers=args.test_nlayers,
                            nclass=data.nclass, device=device).to(device)

        # model = GraphSage(nfeat=features.shape[1],
        #                   nhid=args.student_hidden,
        #                   nclass=data.nclass,
        #                   nlayers=args.student_nlayers,
        #                   dropout=args.student_dropout,
        #                   device=args.device
        #                   ).to(device)



        best_acc_val = best_acc_test = best_acc_it = 0
        best_val_auc = best_test_auc = 0
        my_best_val_auc = my_best_test_auc =0
        train_iters = args.test_model_iters
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for i in range(train_iters):

            model.train()
            optimizer.zero_grad()
            # _, output_syn = model.forward(feat_syn, adj_syn_eval)
            _, output_syn = model.forward(feat_syn, adj_init)

            loss_train = F.nll_loss(output_syn[c_idx_target], labels_syn[c_idx_target])  # 标签类别数跟预测类别数不同

            loss_train.backward()
            optimizer.step()

            if i % 20 == 0:

                model.eval()
                labels_test = torch.LongTensor(data.labels_test).to(device)
                labels_val = torch.LongTensor(data.labels_val).to(device)

                if args.dataset in ['DGraphFin']:

                    # _, output = model.forward(feat_syn, adj_syn_eval)
                    _, output = model.forward(feat_syn, adj_init)

                    val_auc = metrics.roc_auc_score(labels_init[idx_target_val].cpu().numpy(),
                                                    output[idx_target_val, 1].detach().cpu().numpy())
                    test_auc = metrics.roc_auc_score(labels_init[idx_target_test].cpu().numpy(),
                                                     output[idx_target_test, 1].detach().cpu().numpy())

                    # print("验证集auc：", val_auc)
                    # print("测试集auc：", test_auc)

                    if test_auc > best_test_auc:
                        best_test_auc = test_auc
                        best_val_auc = val_auc
                    if best_test_auc > my_best_test_auc:
                        my_best_val_auc = best_val_auc
                        my_best_test_auc = best_test_auc


                    logging.info('测试：now val auc:{},now test auc:{}'.format(my_best_val_auc, my_best_test_auc))


                else:
                    # Full graph
                    adj, feature = utils.to_tensor(data.adj_full, data.feat_full)
                    adj = adj.to(device)
                    feature = feature.to(device)
                    _, output = model.predict(feat_syn, adj_syn)
                    acc_val = utils.accuracy(output[idx_target_tra], labels_val)
                    acc_test = utils.accuracy(output[data.idx_test], labels_test)

                    if acc_val.item() > best_acc_val:
                        best_acc_val = acc_val.item()
                        best_acc_test = acc_test.item()
                        best_acc_it = i

        logging.info('FINAL BEST ACC TEST: {:.6f} with in {}-iteration'.format(best_acc_test, best_acc_it))
        return best_val_auc, best_test_auc

    def distill(self, writer):

        args = self.args  # 获取参数列表
        data = self.data  # 获取所有数据
        features, adj, labels = data.feat_full, data.adj_full, data.labels_full  # 整个数据集特征 邻接矩阵 标签
        (feat_init, adj_init, labels_init, idx_select_train, idx_target_tra, idx_target_val, idx_target_test,
         idx_target, idx_back, c_idx_target, c_idx_back)= self.get_coreset_init(features, adj, labels)  # 获取初始训练特征 邻接矩阵 标签

        adj_syn_init = adj_init
        feat_init, adj_init, labels_init = utils.to_tensor(feat_init, adj_init, labels_init, device=self.device)
        features_tensor, adj_tensor, labels_tensor = utils.to_tensor(features, adj, labels, device=self.device)
        # adj_tensor_norm = utils.normalize_adj_tensor(adj_tensor, sparse=True)  # 归一化原始邻接矩阵
        adj_syn_input = adj_init
        # edge_idx = np.array(list(product(c_idx_target, c_idx_back)))
        edge_idx = adj_to_edge_matrix_sparse(adj_syn_init)
        edge_idx = np.array(edge_idx)
        # 计算原始图的类中心
        ori_unique_labels = np.unique(labels)
        ori_class_centers = {}
        for label in ori_unique_labels:
            class_indices = np.where(labels == label)[0]  # 获取该类的所有索引
            class_indices = torch.tensor(class_indices)
            class_center = (torch.tensor(features)[class_indices]).mean(dim=0)  # 计算该类的类中心（特征向量均值）
            ori_class_centers[label] = class_center

        # 特征聚合
        feat_agg = aggregate_features_without_network(feat_init, adj_init, add_self_loop=True)

        last_class_centers = None  # 保存上一轮的类中心

        self.feat_syn.data.copy_(feat_agg)

        self.labels_syn = labels_init
        # self.adj_syn_init = adj_init
        # self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        file_idx, expert_idx, expert_files = self.expert_load()

        self.labels_syn = labels_init

        self.syn_lr = torch.tensor(args.lr_student).to(self.device)

        if args.optim_lr == 1:
            self.syn_lr = self.syn_lr.detach().to(self.device).requires_grad_(True)
            if args.optimizer_lr == 'Adam':
                optimizer_lr = torch.optim.Adam([self.syn_lr], lr=args.lr_lr)
            elif args.optimizer_lr == 'SGD':
                optimizer_lr = torch.optim.SGD([self.syn_lr], lr=args.lr_lr, momentum=0.1)
                # optimizer_lr = torch.optim.SGD([self.syn_lr], lr=args.lr_lr)


        eval_it_pool = np.arange(0, args.ITER + 1, args.eval_interval).tolist()  # 生成一个迭代次数大小的序列
        model_eval_pool = get_eval_pool(args.eval_type, args.condense_model, args.eval_model)  # GCN
        accs_all_exps = dict()  # record performances of all experiments
        for key in model_eval_pool:
            accs_all_exps[key] = []

        best_accs_test = {m: 0 for m in model_eval_pool}
        best_accs_test_iter = {m: 0 for m in model_eval_pool}
        best_model_std_test = {m: 0 for m in model_eval_pool}

        best_loss = 1.0
        best_loss_it = 0
        adj_syn_norm_key = {'0': 0}
        best_test_auc = best_val_auc = my_best_val_auc = my_best_test_auc = 0
        best_test_acc = best_val_acc = my_best_val_acc = my_best_test_acc = 0
        for it in range(0, args.ITER + 1):  # 迭代401轮
            # logging.info(adj_syn_norm_key['0'])
            model = GCN(nfeat=data.feat_train.shape[1], nhid=args.student_hidden,
                        nclass=data.nclass, dropout=args.student_dropout, nlayers=args.student_nlayers,
                        device=self.device).to(self.device)
            # model.initialize()
            # model = GraphSage(nfeat=features.shape[1],
            #                   nhid=args.student_hidden,
            #                   nclass=data.nclass,
            #                   nlayers=args.student_nlayers,
            #                   dropout=args.student_dropout,
            #                   device=args.device
            #                   ).to(device)
            model = ReparamModule(model)

            model.train()

            num_params = sum([np.prod(p.size()) for p in (model.parameters())])  # 43304 当前网络所有参数数量

            if args.load_all:  # 是否使用所有专家路径
                expert_trajectory = self.buffer[np.random.randint(0, len(self.buffer))]
            else:
                expert_trajectory = self.buffer[expert_idx]  # 选取第0个参数列表 (共10个 每个31 )
                expert_idx += 1  # 索引数+1
                if expert_idx == len(self.buffer):  # 当选择到最后一个专家轨迹时
                    expert_idx = 0
                    file_idx += 1
                    if file_idx == len(expert_files):
                        file_idx = 0
                        random.shuffle(expert_files)
                    print("loading file {}".format(expert_files[file_idx]))
                    if args.max_files != 1:
                        del self.buffer
                        self.buffer = torch.load(expert_files[file_idx])
                    if args.max_experts is not None:
                        self.buffer = self.buffer[:args.max_experts]
                    random.shuffle(self.buffer)

            if args.expanding_window:  # 扩展窗口 选择当前匹配参数的范围
                Upper_Bound = args.max_start_epoch_s + it
                Upper_Bound = min(Upper_Bound, args.max_start_epoch)
            else:
                Upper_Bound = args.max_start_epoch


            np.random.seed(it)
            # 选择当前开始的参数列表
            start_epoch = np.random.randint(args.min_start_epoch, Upper_Bound)  # Upper_Bound最大为30 start_epoch最大也为30

            np.random.seed(args.seed_student)

            start_epoch = start_epoch // 10  # 最大为3
            starting_params = expert_trajectory[start_epoch]  # 最开始时的参数列表

            if args.interval_buffer == 1:
                # print(start_epoch + args.expert_epochs // 10)
                target_params = expert_trajectory[start_epoch + args.expert_epochs // 10]  # 目标参数列表
                # print(start_epoch + args.expert_epochs // 10)
            else:
                target_params = expert_trajectory[start_epoch + args.expert_epochs]

            target_params = torch.cat([p.data.to(self.device).reshape(-1) for p in target_params], 0)  # 所有层参数叠加

            student_params = [
                torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0).requires_grad_(
                    True)]  # 学生模型最开始的参数轨迹

            starting_params = torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0)

            param_loss_list = []  # 参数损失列表
            param_dist_list = []  # 参数距离列表
            logging.info('it:{}--feat_max = {:.4f}, feat_min = {:.4f}'.format(it, torch.max(self.feat_syn),
                                                                              torch.min(self.feat_syn)))

            feat_syn = self.feat_syn
            now_feat = feat_syn.data.clone().detach()


            # size = self.feat_syn.shape[0]
            # indices = torch.arange(size).repeat(2, 1).to(self.device)  # 对角线的行列索引
            # values = torch.ones(size, device=self.device)  # 对角线的值为 1
            # adj_syn_eval = torch.sparse_coo_tensor(indices, values, size=(size, size))
            adj_syn_eval = adj_init
            self.adj_syn = adj_init
            # tag
            # self.adj_syn = adj_syn_input
            for step in range(args.syn_steps):
                forward_params = student_params[-1]  # 使用第一个参数列表进行分类
                feat_out, output_syn = model.forward(feat_syn, adj_syn_eval,
                                                     flat_param=forward_params)  # ReparamModule 得到第一层嵌入表示和预测标签分布
                loss_syn = F.nll_loss(output_syn[idx_target_tra], self.labels_syn[idx_target_tra])

                grad = torch.autograd.grad(loss_syn, student_params[-1], create_graph=True)[0]

                if args.soft_label == True:
                    acc_syn = utils.accuracy(output_syn[idx_target_tra],
                                             torch.argmax(self.labels_syn[idx_target_tra], dim=1))  # 计算准确率
                else:
                    acc_syn = utils.accuracy(output_syn[idx_target_tra], self.labels_syn[idx_target_tra])  # 计算准确率
                # student_params.append(student_params[-1] - self.syn_lr * grad)
                student_params[-1] = student_params[-1] - self.syn_lr * grad  # 学生模型参数进行梯度下降



                if step % 10 == 0:
                    _, output_test = model.forward(feat_syn, adj_syn_eval, flat_param=student_params[-1])

                    val_auc = metrics.roc_auc_score(labels_init[idx_target_val].cpu().numpy(),
                                                    output_test[idx_target_val, 1].detach().cpu().numpy())
                    test_auc = metrics.roc_auc_score(labels_init[idx_target_test].cpu().numpy(),
                                                     output_test[idx_target_test, 1].detach().cpu().numpy())
                    # val_acc = utils.accuracy(output_test[idx_target_val], labels_init[idx_target_val])
                    # test_acc = utils.accuracy(output_test[idx_target_test], labels_init[idx_target_test])

                    # print("验证集auc：", val_auc)
                    # print("测试集auc：", test_auc)
                    # print("验证集acc：", val_acc)
                    # print("测试集acc：", test_acc)
                    if val_auc > best_val_auc:  # 记录最好验证集准确率、测试准确率、以及迭代轮次
                        best_val_auc = val_auc
                        best_test_auc = test_auc
                        best_it = it

                    logging.info('now val auc:{},now test auc:{}'.format(val_auc, test_auc))

                    logging.info('now best val auc:{},now best test auc:{},now best iter:{},'.format(best_val_auc, best_test_auc, best_it))
                    logging.info('loss = {:.4f},acc_syn = {:.4f},acc_test = {:.4f}'.format(loss_syn.item(),
                                                                                           acc_syn.item(),
                                                                                           best_test_auc.item()))

            # 分布匹配损失

            unique_labels = np.unique(labels_init.cpu().numpy())

            class_centers = {}
            for label in unique_labels:
                class_indices = np.where(labels_init.cpu().numpy() == label)[0]  # 获取该类的所有索引
                class_indices = torch.tensor(class_indices)
                with torch.no_grad():
                    class_center = (now_feat[class_indices]).mean(dim=0) # 计算该类的类中心（特征向量均值）
                    class_centers[label] = class_center
            acculate_dis = torch.tensor(0.0, device=self.device, requires_grad=True)  # 初始化时加入梯度

            if last_class_centers is not None and it % 10 == 0:
                history_loss = 0.0
                count = 0
                for label in class_centers:
                    if label in last_class_centers:
                        # 计算两个类中心之间的距离（如L2距离）
                        dist = torch.norm(class_centers[label] - last_class_centers[label], p=2)
                        history_loss += dist
                        count += 1

                if count > 0:
                    history_loss = history_loss / count  # 平均距离

            else:
                history_loss = 0.0
                # 更新历史分布信息
            last_class_centers = class_centers.copy()  # 深拷贝避免引用问题

            for i in ori_class_centers.keys():
                ori_class_centers[i] = ori_class_centers[i].to(self.device)
                dis_rate = torch.tensor(np.where(labels_init.cpu().numpy() == i)[0]).shape[0]/labels_init.shape[0]
                acculate_dis = acculate_dis + dis_rate*(ori_class_centers[i] - class_centers[i])
            dis_loss = torch.sum(acculate_dis ** 2)

            # 轨迹匹配损失
            param_loss = torch.tensor(0.0).to(self.device)  # 记录累计参数损失
            param_dist = torch.tensor(0.0).to(self.device)  # 记录累计迭代损失

            param_loss += torch.norm(student_params[-1] - target_params, 2)  # 计算迭代一百轮后学生模型与教师模型的参数损失
            param_dist += torch.norm(starting_params - target_params, 2)  # 计算开始参数与目标参数的一个损失

            # param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
            param_loss = param_loss
            param_loss_list.append(param_loss)  # 存储当前损失
            param_dist_list.append(param_dist)

            param_loss = 0.5 * param_loss

            param_loss /= num_params  # 损失值平均到每个参数
            param_dist /= num_params


            param_loss /= param_dist

            grand_loss = param_loss  # 损失LM


            # total_loss = adaptive_loss(grand_loss, dis_loss)

            total_loss = grand_loss + args.beta * dis_loss + history_loss # 记录当前总的损失


            nc_feat = self.feat_syn[c_idx_target].clone().detach()

            self.optimizer_feat.zero_grad()
            # self.optimizer_pge.zero_grad()

            total_loss.backward()
            if it % 50 < 20:
                self.optimizer_feat.step()

            with torch.no_grad():
                self.feat_syn[c_idx_target] = nc_feat



            if it % 1 == 0:
                logging.info(
                    "Iteration {}: Total_Loss = {:.4f}, Grand_Loss={:.4f},dis_Loss={:.4f}, his_loss= {}, Student_LR = {:6f}".format(
                        it,
                        total_loss.item(),

                        grand_loss.item(),
                        dis_loss.item(),
                        history_loss,
                        self.syn_lr.item()))


            if it in eval_it_pool and it > 0:
                for model_eval in model_eval_pool:
                    logging.info(
                        'Evaluation: model_train = {}, model_eval = {}, iteration = {}'.format(args.condense_model,
                                                                                               model_eval,
                                                                                               it))

                    best_acc_eval, best_acc_test = self.eval_synset(args)
                    feat_syn_save, adj_syn_save, label_syn_save = self.synset_save()

                    best_acc_eval = np.array(best_acc_eval)
                    best_acc_test = np.array(best_acc_test)

                    best_acc_eval = np.mean(best_acc_eval)
                    best_acc_test = np.mean(best_acc_test)
                    # best_acc_eval = np.max(best_acc_eval)
                    # best_acc_test = np.max(best_acc_test)
                    acc_eval_std = np.std(best_acc_eval)
                    # writer.add_scalar('ntk_acc_eval_curve', best_acc_eval, it)

                    if best_acc_eval > best_accs_test[model_eval]:
                        best_accs_test[model_eval] = best_acc_eval
                        best_accs_test_iter[model_eval] = it
                        torch.save(adj_syn_save,
                                   f'{args.log_dir}/adj_{args.dataset}_{args.reduction_rate}_best_eval_acc_{args.seed_student}_ours.pt')
                        torch.save(feat_syn_save,
                                   f'{args.log_dir}/feat_{args.dataset}_{args.reduction_rate}_best_eval_acc_{args.seed_student}_ours.pt')
                        torch.save(label_syn_save,
                                   f'{args.log_dir}/label_{args.dataset}_{args.reduction_rate}_best_eval_acc_{args.seed_student}_ours.pt')

                        logging.info(
                            'now best test_acc occurs, the eval_acc is {:.4f}, the test_acc is {:.4f}, the iter is {:.2f} ! ! !'.format(
                                best_acc_eval * 100.0, best_acc_test * 100.0, it))


            writer.add_scalar('grand_loss_curve', grand_loss.item(), it)
            # torch.cuda.empty_cache()

            # gc.collect()

        for model_eval in model_eval_pool:
            logging.info('Evaluation ACC: {} for best eval_acc = {:.5f}, within {} iter'.format(
                model_eval, best_accs_test[model_eval], best_accs_test_iter[model_eval]))

        logging.info('This is the smallest loss: {:.06f} within {} iteration'.format(best_loss, best_loss_it))

    def get_coreset_init(self, features, adj, labels):
        logging.info('Loading from: {}'.format(self.args.coreset_init_path))
        idx_selected_train = np.load(
            f'{self.args.coreset_init_path}/idx_{self.args.dataset}_{self.args.reduction_rate}_{self.args.coreset_method}_{self.args.coreset_seed}.npy')

        # idx_target = idx_selected_train[self.data.idx_train]
        idx_selected_train = np.unique(idx_selected_train)
        idx_target_tra = np.searchsorted(idx_selected_train, self.data.idx_train)  # 目标节点对应到已选择训练节点索引中的位置
        idx_target_val = np.searchsorted(idx_selected_train, self.data.idx_val)
        idx_target_test = np.searchsorted(idx_selected_train, self.data.idx_test)
        idx_target = np.concatenate((self.data.idx_train, self.data.idx_val, self.data.idx_test), axis=0)
        idx_back = np.setdiff1d(idx_selected_train, idx_target)

        c_idx_target = np.concatenate((idx_target_tra, idx_target_val, idx_target_test), axis=0)
        c_idx_back = np.searchsorted(idx_selected_train, idx_back)
        # idx_selected_train = np.load(
        #         f'{self.args.coreset_init_path}')
        feat_train = features[idx_selected_train]  # 所选训练集特征
        adj_train = adj[np.ix_(idx_selected_train, idx_selected_train)]  # 训练集结构
        labels_train = labels[idx_selected_train]  # 训练集标签
        return (feat_train, adj_train, labels_train, idx_selected_train, idx_target_tra,
                idx_target_val, idx_target_test, idx_target, idx_back, c_idx_target, c_idx_back)




def one_hot(x,
            num_classes,
            center=True,
            dtype=np.float32):
    assert len(x.shape) == 1
    one_hot_vectors = np.array(x[:, None] == np.arange(num_classes), dtype)
    if center:
        one_hot_vectors = one_hot_vectors - 1. / num_classes
    return one_hot_vectors


def calc(gntk, feat1, feat2, diag1, diag2, A1, A2):
    return gntk.gntk(feat1, feat2, diag1, diag2, A1, A2)


def loss_acc_fn_train(data, k_ss, k_ts, y_support, y_target, reg=5e-2):
    # print(k_ss.device, torch.abs(torch.tensor(reg)).to(k_ss.device),torch.trace(k_ss).device, torch.eye(k_ss.shape[0]).device)
    k_ss_reg = (k_ss + torch.abs(torch.tensor(reg)).to(k_ss.device) * torch.trace(k_ss).to(k_ss.device) * torch.eye(
        k_ss.shape[0]).to(k_ss.device) / k_ss.shape[0])
    pred = torch.matmul(k_ts[data.idx_train, :].cuda(), torch.matmul(torch.linalg.inv(k_ss_reg).cuda(),
                                                                     torch.from_numpy(y_support).to(
                                                                         torch.float64).cuda()))
    mse_loss = torch.nn.functional.mse_loss(pred.to(torch.float64).cuda(),
                                            torch.from_numpy(y_target).to(torch.float64).cuda(), reduction="mean")
    acc = 0
    return mse_loss, acc


def loss_acc_fn_eval(data, k_ss, k_ts, y_support, y_target, reg=5e-2):
    k_ss_reg = (k_ss + np.abs(reg) * np.trace(k_ss) * np.eye(k_ss.shape[0]) / k_ss.shape[0])
    pred = np.dot(k_ts, np.linalg.inv(k_ss_reg).dot(y_support))
    mse_loss = 0.5 * np.mean((pred - y_target) ** 2)
    acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y_target, axis=1))
    return mse_loss, acc


def adj_to_edge_matrix_sparse(adj_init):
    # 将邻接矩阵转换为稀疏格式
    adj_matrix = sp.csr_matrix(adj_init)

    # 获取非零元素的行和列索引
    rows, cols = adj_matrix.nonzero()

    # 对于无向图，避免重复边
    edges = []
    for i, j in zip(rows, cols):
        if i < j:
            edges.append([i, j])
        elif i > j:
            edges.append([j, i])

    return edges


log_w1 = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
log_w2 = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))

def adaptive_loss(loss1, loss2):

    # 计算指数权重
    w1, w2 = torch.exp(log_w1), torch.exp(log_w2)
    # 使用 softmax 归一化，确保总和为1
    weights = torch.softmax(torch.tensor([w1.log(), w2.log()]), dim=0)
    return weights[0] * loss1 + weights[1] * loss2