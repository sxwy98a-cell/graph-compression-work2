# Graph Compression Work 2

面向节点分类任务的图压缩与图凝缩实验代码。项目包含教师模型训练轨迹生成、节点子集初始化、基于参数轨迹匹配的凝缩，以及凝缩过程中的模型评估。

当前主要实验入口为 `buffer_transduct_cl.py` 和 `my_condense.py`，凝缩逻辑位于 `my_geomtt.py`。仓库保存代码与实验配置；原始数据、教师轨迹和训练结果需要在本地准备。

> 当前仓库是实验代码快照，尚未提供锁定版本的运行环境，且部分模块与历史脚本缺失。下文命令用于说明实验流程和参数衔接，完整训练尚未在此仓库环境中验证；请先完成“运行前准备”。

## 目录结构

```text
.
├── buffer_transduct_cl.py        # 教师训练与参数轨迹保存
├── my_condense.py               # 凝缩入口：读取配置、加载数据、启动训练
├── my_geomtt.py                 # MetaGtt：初始化、轨迹匹配、评估与保存
├── train_coreset.py             # 节点子集选择与基线训练入口
├── get_data.py                  # IMDB、DBLP、ogbn-mag、elliptic 数据读取
├── utils.py                     # 数据转换、训练调度与图处理工具
├── utils_graphsaint.py          # GraphSAINT 格式与 DGraphFin 数据加载
├── models/                     # GCN、GAT、GraphSAGE、SGC 等模型及辅助模块
├── scripts/
│   ├── buffer.sh               # 历史教师训练命令
│   ├── condensation.sh         # 历史凝缩命令
│   ├── coreset.sh              # 历史节点选择命令
│   ├── evaluation.sh           # 历史评估命令与记录
│   └── config/
│       ├── config_distill.json # 凝缩实验配置
│       └── config_test.json    # 历史评估配置
├── data/                       # 本地数据，Git 忽略
└── logs/                       # 实验日志与输出，Git 忽略
```

`data/` 和 `logs/` 在运行时按需准备或生成，不包含在 GitHub 仓库中。

## 环境与依赖

代码使用 Python、PyTorch 和 PyTorch Geometric，并依赖 CUDA 运算。建议使用独立环境；项目尚未固定 Python、PyTorch、PyG 和 CUDA 的版本，应优先沿用原始实验环境。

主要第三方依赖如下：

| 类别 | 依赖 |
| --- | --- |
| 深度学习与图计算 | `torch`、`torchvision`、`torch_geometric`、`torch_sparse` |
| 数值计算与数据处理 | `numpy`、`scipy`、`pandas`、`scikit-learn` |
| 图数据与工具 | `deeprobust`、`ogb` |
| 日志与进度 | `tensorboardX`、`tensorboard`、`tqdm`、`wandb` |
| 可选显存日志 | `pynvml`，用于 `models/gpu_memory_log.py` |

先根据操作系统与 CUDA 环境安装 PyTorch / torchvision，再安装与其版本匹配的 PyG 和稀疏扩展。安装说明见 [PyTorch 官方指南](https://pytorch.org/get-started/locally/) 和 [PyG 官方指南](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。本项目直接导入 `torch_sparse`，需要安装该扩展。

其余依赖可在选定的实验环境中安装：

```bash
python -m pip install numpy scipy pandas scikit-learn tqdm deeprobust ogb tensorboardX tensorboard wandb
```

代码使用了 `torch_geometric.data.NeighborSampler`、`train_test_split_edges` 等接口；安装后需要核对所选 PyG 版本是否兼容。

## 运行前准备

### 1. 获取代码

```bash
git clone https://github.com/sxwy98a-cell/graph-compression-work2.git
cd graph-compression-work2
```

以下命令均在仓库根目录执行，多行示例采用 Bash 语法。Windows 上可使用 WSL 或 Git Bash；PowerShell 中可将命令合并为一行。

### 2. 准备数据

教师训练和凝缩入口目前按以下分支读取数据：

| 数据集 | 入口中的读取方式 | 本地准备要求 |
| --- | --- | --- |
| `cora`、`citeseer`、`pubmed` | `utils.get_dataset()` / PyG Planetoid | 数据保存在 `data/<dataset>/`；首次加载可能下载 |
| `ogbn-arxiv`、`flickr`、`reddit` | `DataGraphSAINT` | 提供预处理后的 GraphSAINT 格式文件 |
| `DGraphFin` | `DgraphFinData` | 提供 `data/DGraphFin/dgraphfin.npz` |

GraphSAINT 格式目录至少需要：

```text
data/<dataset>/
├── adj_full.npz     # SciPy 稀疏邻接矩阵
├── feats.npy        # 节点特征
├── role.json        # tr、va、te 对应训练、验证、测试节点索引
└── class_map.json   # 节点标签映射
```

DGraphFin 加载器读取 NPZ 中的 `x`、`y`、`edge_index`、`train_mask`、`valid_mask`、`test_mask`。当前代码将后三项直接作为节点索引使用；如果数据提供的是布尔掩码，需要先转换为索引数组。

`get_data.py` 另有 IMDB、DBLP、ogbn-mag 和 elliptic 的读取逻辑，但包含原实验服务器的绝对路径，需要替换为自己的数据路径。它们尚未接入 `my_condense.py` 的对应数据分支，配置文件中的同名配置也不能直接视为可运行实验。

### 3. 核对 GPU 与路径

- `DgraphFinData` 中的标签加载固定使用 `cuda:1`。下文 DGraphFin 示例沿用该设备；只有一张 GPU 时，需要先调整加载器中的设备设置，再统一入口参数。仅修改 `--device` 无法覆盖该硬编码。
- `my_condense.py` 的默认配置路径为原服务器绝对路径，运行时请显式传入 `--config scripts/config/config_distill.json`。
- 某些配置节含 `device` 字段；配置在命令行解析之后加载，同名字段会覆盖命令行参数。请一并核对所选配置。

### 4. 准备节点初始化索引

`MetaGtt.distill()` 会先加载如下文件：

```text
<coreset_init_path>/idx_<dataset>_<reduction_rate>_<coreset_method>_<coreset_seed>.npy
```

例如，`DGraphFin-r015 ` 配置搭配默认 `coreset_method=random`、`coreset_seed=15` 时，预期文件是：

```text
logs/Coreset/DGraphFin-reduce_0.15-coreinit/idx_DGraphFin_0.15_random_15.npy
```

可以提供原实验生成的索引，或补齐节点选择流程后重新生成。当前 `train_coreset.py` 导入 `from coreset import KCenter, Herding, Random`，但仓库尚未包含 `coreset.py` 或 `coreset/` 包，运行该脚本前需要补齐模块。

还需对齐两处节点划分：`my_condense.py` 将训练、验证和测试节点全部作为目标节点，其余为背景节点；`train_coreset.py` 当前将验证和测试节点作为目标节点、训练节点作为背景节点。初始化索引需要包含凝缩流程使用的全部目标节点，选取的背景节点数量也应与凝缩特征张量大小一致。

## 实验流程

### 1. 生成教师参数轨迹

完成环境和数据准备后，以 DGraphFin 为例：

```bash
python buffer_transduct_cl.py \
  --dataset DGraphFin \
  --device cuda:1 \
  --buffer_model_type GCN \
  --teacher_nlayers 3 \
  --teacher_hidden 256 \
  --teacher_epochs 1000 \
  --num_experts 10 \
  --traj_save_interval 10 \
  --param_save_interval 10 \
  --lr_teacher 0.01 \
  --optim Adam \
  --lam 0.85 \
  --T 800 \
  --scheduler root
```

轨迹写入：

```text
logs/Buffer/DGraphFin-lam-0.85-T-800-scheduler-root/replay_buffer_0.pt
```

`param_save_interval` 控制参数快照保存间隔，`traj_save_interval` 控制每个文件保存的专家轨迹数量。当前代码仅在轨迹数量达到 `traj_save_interval` 时写入文件，结束时不会另存不足一组的剩余轨迹；请让 `num_experts` 与该间隔匹配。

### 2. 运行凝缩

确认教师轨迹、初始化索引和配置都已准备后：

```bash
python my_condense.py \
  --config scripts/config/config_distill.json \
  --section "DGraphFin-r015 " \
  --device cuda:1 \
  --student_nlayers 3 \
  --student_hidden 256 \
  --lam 0.85 \
  --T 800 \
  --scheduler root
```

注意 `"DGraphFin-r015 "` 的结尾有一个空格。当前 JSON 的许多配置节名称带尾随空格，需要按实际键名传入，否则配置读取会失败。

可用以下命令查看准确键名：

```bash
python -c "import json; c=json.load(open('scripts/config/config_distill.json', encoding='utf-8')); print('\n'.join(repr(k) for k in c))"
```

教师阶段和凝缩阶段的 `dataset`、`lam`、`T`、`scheduler` 应保持一致，因为凝缩入口会用它们重新拼接轨迹路径：

```text
./logs/Buffer/<dataset>-lam-<lam>-T-<T>-scheduler-<scheduler>
```

该路径会覆盖 JSON 中的 `buffer_path` 字段，教师阶段也应使用默认的 `logs` 输出根目录，或同步调整凝缩入口。

### 3. 查看输出

凝缩运行目录位于 `logs/Distill/`，包含 `train.log` 和 `tbx_log/`。评估结果改进时会保存：

```text
adj_<dataset>_<reduction_rate>_best_eval_acc_<seed_student>_ours.pt
feat_<dataset>_<reduction_rate>_best_eval_acc_<seed_student>_ours.pt
label_<dataset>_<reduction_rate>_best_eval_acc_<seed_student>_ours.pt
```

可查看 TensorBoard 日志：

```bash
tensorboard --logdir logs
```

## 主要参数

| 参数 | 含义 |
| --- | --- |
| `--config` / `--section` | 凝缩配置文件与配置节名称 |
| `--device` | 入口使用的计算设备，还需核对配置覆盖与硬编码 |
| `--num_experts` / `--teacher_epochs` | 专家数量与每个教师的训练轮数 |
| `--lam` / `--T` / `--scheduler` | 教师训练调度参数；调度方式含 `linear`、`root`、`geom` |
| `reduction_rate` | 背景节点压缩比例，在凝缩配置中设置 |
| `coreset_init_path` | 初始化索引文件所在目录，在凝缩配置中设置 |
| `--coreset_method` / `--coreset_seed` | 初始化索引文件名中的选择方法与种子 |
| `--expert_epochs` / `--syn_steps` | 轨迹匹配相关的专家跨度与合成训练步数 |
| `--lr_feat` | 凝缩特征学习率 |
| `ITER` / `eval_interval` | 凝缩迭代上限与评估间隔，在配置中设置 |
| `--load_all` | 一次加载全部专家轨迹；需要足够内存 |

当前凝缩节点数按 `int(背景节点数 × reduction_rate + 目标节点数)` 计算；`reduction_rate` 不是整个图最终保留节点的比例。

## 当前仓库的复现范围

- `scripts/` 保留历史命令。其中引用的 `buffer_inductive_cl.py`、`condense_transduct_1.py`、`condense_transduct_2.py`、`eval_condg.py` 等文件未包含在当前仓库，不宜直接整批运行这些脚本。
- `config_test.json` 保留历史评估配置，当前未包含对应独立评估入口；凝缩过程中的评估位于 `my_geomtt.py`。
- 数据、初始化索引、教师轨迹和已训练模型均未上传；需要自行准备。
- 仓库尚未提供经验证的完整复现环境、统一实验结果表或论文引用信息。历史脚本中的结果注释应结合原始实验记录核实。

`.gitignore` 已排除数据目录、日志、模型权重、Python 缓存和本地环境文件。
