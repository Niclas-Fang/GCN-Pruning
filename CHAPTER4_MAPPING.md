# 《基于 Shapley 值的神经网络剪枝研究》第四章与 GCN_pruning 代码对应及复现指南

> **论文**：哈尔滨工业大学硕士学位论文《基于 Shapley 值的神经网络剪枝研究》，王澪彬，2026 年 3 月，导师张达治。
>
> **第四章**：基于 Shapley 值的 GNN 剪枝研究（正文第 38–58 页）。
>
> **代码版本**：`GCN_pruning`，git `main` 分支，提交 `2bf7ff1`（工作区干净）；跨目录部分涉及 `../GNNShap`。
>
> **复现环境**：`/home/niclas/shapley/.venv`，Python 3.12，PyTorch 2.11.0+cu128，PyTorch Geometric 2.8.1。
>
> **本文档目标**：把第四章的每一节、每一个公式、每一张表/图对应到具体代码文件和函数，并给出能够复现实验结果的命令；对无法逐位复现的部分明确标注原因和边界。

---

## 0. 阅读说明

### 0.1 复现等级

本文档用四个标记描述每个论文元素与代码的对应程度：

| 标记 | 含义 |
|---|---|
| ✅ | **可精确复现**：用现有代码/结果文件可以逐位得到论文数值。 |
| 🟡 | **可复现方法路径，数值近似**：代码实现了同一算法，但论文使用的具体随机种子、toy 图版本或价值函数细节未完整保留，数值只能定性一致。 |
| 🔵 | **仓库中有结果文件，但当前脚本不直接生成**：例如旧版实验脚本已被改写，结果仍保存在 `outputs/` 或 `GNNShap/*.json`。 |
| ❌ | **当前仓库缺失**：没有对应实现或没有可核对的结果文件。 |

### 0.2 两套实验设置（非常重要）

论文第四章实际上混用了两套完全不同的基座模型。把它们的数值直接放在一起比较会得到错误结论。

| 设置 | 用于 | 数据集 | 模型 | 基线测试精度 |
|---|---|---|---|---|
| **A. GNNShap 设置** | 表 4-3、表 4-4，图 4-8～4-11 | Cora | GCN，2 层，hidden=16，dropout=0.5，lr=0.01 | **0.815** |
| **B. GCN_pruning 设置** | 表 4-5、表 4-6，图 4-12 | Cora | GCN，3 层 GCN + 1 层 MLP，hidden=64，dropout=0.5，lr=0.01，wd=5e-4，200 epochs | **0.7707**（3 seeds 平均） |

- 设置 A 的来源：`../GNNShap/dataset/configs.py` 中 `Cora` 配置（hidden 16、2 层），以及预训练模型 `../GNNShap/pretrained/Cora_pretrained.pt`。
- 设置 B 的来源：`run_large_scale_experiments.py` 的 `train_base_model()`（第 77–113 行），实际配置为 `hidden_dim=64`、`num_gcn_layers=3`、`num_mlp_layers=1`、`dropout=0.5`。
- **论文 4.4.1 节写的是“标准两层 GCN”，但设置 B 的代码实际是 3 层 GCN + 1 层 MLP。** 表 4-5/4-6 的 0.7707 与代码完全对应；表 4-3/4-4 的 0.815 与 GNNShap 的两层 GCN 对应。复现时不要混用。

### 0.3 命令约定

- 除特别说明外，命令的工作目录是：

```bash
cd /home/niclas/shapley/GCN_pruning
```

- Python 解释器使用：

```bash
../.venv/bin/python
```

- **toy 脚本（`toy_model*.py`）不要加 `-I`**。这些脚本依赖脚本所在目录进入 `sys.path` 来导入本地 `scorers`、`coalition_builder`；`-I` 会移除该路径，导致 `ModuleNotFoundError: No module named 'scorers'`。
- `run_large_scale_experiments.py`、`run_dataset_benchmark.py` 等入口会自己把项目根目录插入 `sys.path`，加不加 `-I` 都可以。
- 跨目录的 GNNShap 命令需要：

```bash
cd /home/niclas/shapley/GNNShap
../.venv/bin/python ...
```

---

## 1. 总览：第四章 → 代码地图

```text
第4章 基于 Shapley 值的 GNN 剪枝研究
│
├─ 4.1 图结构剪枝的特点与挑战
│   ├─ 4.1.1 节点删除 / 节点特征屏蔽 / 节点消息屏蔽 / 边掩码
│   │        → pruning/graph_mask.py
│   │        → sanity_graph_mask.py（掩码行为检查）
│   ├─ 4.1.2 命题 4.1–4.3（局部扰动性、全局扰动性、正负贡献）
│   │        → apply_edge_mask / apply_node_mask + toy 实验的正负 Shapley 值
│   └─ 4.1.3 剪枝对象选择（为什么以边为主）
│            → 理论分析，无独立代码
│
├─ 4.2 图结构剪枝的合作博弈建模
│   ├─ 4.2.1 边作为玩家
│   │        → scorers/shapley.py::CoalitionValueFunction（unit_type='edge'）
│   ├─ 4.2.2 边联盟划分
│   │        → coalition_builder/edge_builders.py
│   └─ 4.2.3 特征函数 v(S) 与 Shapley 公式
│            → CoalitionValueFunction.__call__ / _evaluate_loss
│            → owen_shapley_two_stage / shapley_approximation / ...
│
├─ 4.3 Shapley 用于图结构剪枝的理论合理性
│   ├─ 4.3.1 toy 图构造
│   │        → toy_model2.py / toy_model4.py（7 节点、7 条无向边）
│   │        → toy_model3.py / toy_model5.py（8 节点变体，注意区别）
│   ├─ 4.3.2 简单图结构的剪枝直观分析
│   │        → toy_model2.py / toy_model4.py 的输出
│   ├─ 4.3.3 表 4-1（不同归因方法在 7 条边上的全局评分）
│   │        → scorers/edge_scorers.py + scorers/shapley.py + toy_model2/3
│   └─ 4.3.4 表 4-2（五种联盟划分下的 Owen-Shapley）
│            → toy_model4.py / toy_model5.py + coalition_builder/edge_builders.py
│
└─ 4.4 GNN 剪枝实验结果与分析
    ├─ 4.4.1 实验设置
    │        → data/loader.py、models/、GNNShap/dataset/configs.py
    ├─ 4.4.2 算法 4-1/4-2、表 4-3/4-4、图 4-8～4-11
    │        → GNNShap/（PGE、Saliency、GNNShap、Two-stage Owen + ev_node/ev_edge）
    │        → GCN_pruning/pruning/graph_mask.py（掩码语义）
    └─ 4.4.3 算法 4-3、图 4-12、表 4-5/4-6
             → experiments/run_joint_pruning.py::JointPruningRunner
             → run_large_scale_experiments.py
             → analyze_results.py / analyze2.py / vis4.py
```

---

## 2. 逐节对应

### 2.1 第 4 章引言：GCN 传播（式 4-1、4-2，图 4-1）

| 论文内容 | 代码位置 | 说明 |
|---|---|---|
| 式 (4-1) 图 \(G=(V,E,\mathbf{X})\) | [data/loader.py](data/loader.py)、[models/gcn_layer.py](models/gcn_layer.py#L162) 的 `GraphData` | Planetoid 数据加载，`x`、`edge_index`、`y`、`train/val/test_mask`。 |
| 式 (4-2) \(\mathbf{H}^{(l+1)}=\sigma(\hat{\mathbf{A}}\mathbf{H}^{(l)}\mathbf{W}^{(l)})\) | [models/gcn_layer.py](models/gcn_layer.py#L10) 的 `GCNConv.forward`（第 46 行） | `GCNConv` 执行 \( \hat{A}H W \)。 |
| \(\hat{\mathbf{A}}=\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\) | [models/gcn_layer.py](models/gcn_layer.py#L75) 的 `normalize_adjacency` | 加自环、对称归一化，返回稀疏 COO 邻接。 |
| 图 4-1 GCN 计算流程 | [models/gcn_net.py](models/gcn_net.py#L11) 的 `GCNNet.forward`（第 86 行） | `input_proj → GCN×n → MLP×m → output_head`。 |

**对应关系**：论文的 \(\mathbf{H}^{(0)}=\mathbf{X}\) 对应 `input_proj` 之前的输入；`input_proj` 是代码额外加的线性投影，不参与剪枝；`output_head` 是分类头，也不参与剪枝。

### 2.2 4.1.1 图结构剪枝中的剪枝单元定义（式 4-3～4-13）

论文定义了三种节点剪枝语义 + 一种边剪枝语义。**GCN_pruning 只实现了“节点消息屏蔽”和“边掩码”两种；节点删除和节点特征屏蔽在 GNNShap 中。**

| 论文语义 | 公式 | GCN_pruning 实现 | 复现等级 | 备注 |
|---|---|---|---|---|
| 节点删除（node deletion） | 式 (4-3)(4-4) | ❌ 无 | 🔵 在 `../GNNShap/ev_node.py` | `prune_graph_by_ratio()` 用 `torch_geometric.utils.subgraph` 做诱导子图。 |
| 节点特征屏蔽（node feature mask） | 式 (4-5)(4-6) | ❌ 无 | 🔵 在 `../GNNShap/ev_node.py` | `prune_graph_by_ratio_feature_mask()`，把被剪节点特征置零。 |
| 节点消息屏蔽（node message mask） | 式 (4-7)～(4-9) | ✅ [pruning/graph_mask.py](pruning/graph_mask.py#L104) 的 `apply_node_mask` | ✅ | 删除“至少一个端点是剪枝节点”的边，再给**所有节点**补自环；被剪节点仍可用自身特征预测。 |
| 边掩码（edge mask） | 式 (4-10)～(4-13) | ✅ [pruning/graph_mask.py](pruning/graph_mask.py#L68) 的 `apply_edge_mask` | ✅ | 只保留 `edge_mask=True` 的边，补全无向边并加自环。 |

**关键代码对照**：

```python
# pruning/graph_mask.py
def apply_node_mask(data, node_mask, ensure_undirected=True, preserve_self_loops=True):
    # 只保留两个端点都保留的边
    edge_mask = node_mask[src] & node_mask[dst]
    edge_index_masked = edge_index_no_loops[:, edge_mask]
    # 给所有节点补自环（包括被剪节点）
    edge_index_masked = add_self_loops(edge_index_masked, num_nodes)
```

- 这与论文式 (4-7) 的 \(\mathbf{A}_S=\mathrm{Diag}(\mathbf{m}_S)\mathbf{A}\mathrm{Diag}(\mathbf{m}_S)\) 等价：被剪节点既不向外发送消息，也不接收外来消息；自环保证其仍能做出预测。
- `apply_node_mask` **不是**节点删除：节点数、节点特征、节点索引都不变，因此全局节点分类的输出维度不变。这正是论文 4.1.1 节最后选择“消息屏蔽”作为默认节点剪枝定义的原因。
- `apply_edge_mask` 对应论文的 \(\mathbf{A}_S=\mathbf{A}\odot\mathbf{M}_S\)，再按 GCN 流程加自环并归一化。

**图 4-2～4-6（Cora 2-hop 子图剪枝示意）**：❌ 仓库中没有生成这些示意图的绘图脚本。`sanity_graph_mask.py` 只打印图统计和掩码前后的边数/自环数，不画图。掩码本身可以复现，示意图需要自己用 `networkx`/`matplotlib` 重画。

**可直接运行的掩码检查**：

```bash
cd /home/niclas/shapley/GCN_pruning

# 边掩码 + 节点消息屏蔽 + 组合掩码的前向检查
../.venv/bin/python sanity_graph_mask.py

# 模型通道掩码检查
../.venv/bin/python sanity_model_mask.py

# 基础前向/邻接归一化检查
../.venv/bin/python sanity_forward.py
```

### 2.3 4.1.2 节点剪枝与边剪枝对消息传递的影响（命题 4.1～4.3）

| 论文内容 | 代码/实验对应 | 复现等级 |
|---|---|---|
| 命题 4.1 边剪枝的局部扰动性 | `apply_edge_mask` 只改变被删边端点的度数和归一化系数 | 🟡 理论命题，无单独单元测试；`sanity_graph_mask.py` 可观察边数/度变化 |
| 命题 4.2 节点删除的全局扰动性 | 节点删除不在 GCN_pruning；`../GNNShap/ev_node.py` 的诱导子图会改变所有邻居的度 | 🔵 跨目录 |
| 命题 4.3 边剪枝的正负贡献可分性 | `scorers/shapley.py` 返回带符号的 Shapley 值；`toy_model2.py`/`toy_model4.py` 的输出可直接看到正负号 | 🟡 toy 实验定性验证 |

**说明**：命题 4.1–4.3 是理论分析，仓库没有“命题验证脚本”。能对应的是掩码函数的行为和 toy 图中 Shapley 值的符号模式：
- `toy_model2.py` 的 Exact-SV 输出中，异类连接边 `(0,5)`、`(0,6)` 为负，正类主路径边 `(1,3)`、`(2,4)` 为正；
- `toy_model4.py` 的五种联盟划分下，`(0,5)`、`(0,6)` 与 `(1,3)`、`(2,4)` 的符号模式与论文表 4-2 一致。

### 2.4 4.1.3 图结构剪枝对象的选择分析

这一节是理论论证：节点剪枝语义不唯一、影响范围难控制；边剪枝语义清晰、局部扰动强、能产生正负归因，因此正文以边为主。

- 代码对应：没有单独的“选择分析”脚本。
- 实验对应：表 4-3（节点消息屏蔽）与表 4-4（边删除）的对比；GCN_pruning 的 `JointPruningRunner(graph_unit='node'/'edge')` 也支持两种图单元。
- 复现等级：🟡（实验可复现，论证本身是文字）。

### 2.5 4.2.1 边作为玩家的合作博弈表示

| 论文内容 | 代码位置 |
|---|---|
| 玩家集合 \(\mathcal{E}=E\) | [scorers/shapley.py](scorers/shapley.py#L53) 的 `CoalitionValueFunction`，`unit_type='edge'` |
| 联盟 \(S\subseteq E\) | `CoalitionValueFunction.__call__` 接收边索引列表（第 93 行） |
| 屏蔽后的邻接矩阵 \(\mathbf{A}_S=\mathbf{A}\odot\mathbf{M}_S\) | `_evaluate_edge_coalition`（第 148 行）调用 `apply_edge_mask` |
| 子图模型 \(f_{\mathcal{G}_S}\) | `_evaluate_edge_coalition` 中重建 `adj_norm` 后前向 `MaskedModelWrapper` |
| 式 (4-14) | 对应 `normalize_adjacency(apply_edge_mask(...))` |

关键点：在 GCN_pruning 中，“玩家”是 **`edge_index` 的列索引**（有向边索引），而不是无向边的 `(u,v)` 对。Cora 的 `edge_index` 有 10556 列，因此边 Shapley 的玩家空间是 10556；玩具图有 14 条有向边（7 条无向边）。论文 toy 表 4-1/4-2 报告的是折叠回无向边后的平均值。

### 2.6 4.2.2 图结构中边的联盟划分方式

| 论文划分策略 | 代码类 | 工厂入口 |
|---|---|---|
| 基于源节点（Source-Node） | [SourceNodeEdgeBuilder](coalition_builder/edge_builders.py#L52) | `build_edge_coalitions('source_node', ...)` |
| 基于目标节点（Target-Node） | [TargetNodeEdgeBuilder](coalition_builder/edge_builders.py#L90) | `build_edge_coalitions('target_node', ...)` |
| 基于社区（Community） | [CommunityEdgeBuilder](coalition_builder/edge_builders.py#L127) | `build_edge_coalitions('community', method='louvain', ...)` |
| 基于模体（Motif-Bridge / Motif-Triangle） | [MotifEdgeBuilder](coalition_builder/edge_builders.py#L263) | `build_edge_coalitions('motif_bridge' / 'motif_triangle', ...)` |
| 从节点组推导边组 | [FromNodeGroupsEdgeBuilder](coalition_builder/edge_builders.py#L448) | `build_edge_coalitions('from_node_groups', ...)` |

工厂函数在 [coalition_builder/edge_builders.py](coalition_builder/edge_builders.py#L520)。

论文 4.2.2 节列出的三种策略（Source-Node、Target-Node、Community）和 4.3.4 节表 4-2 的五种策略（Source-Node、Target-Node、Community、Motif-Bridge、Motif-Triangle）在代码中全部有对应。

**可直接运行的联盟构建示例**：

```bash
cd /home/niclas/shapley/GCN_pruning
../.venv/bin/python example_edge_coalitions.py
```

### 2.7 4.2.3 图结构剪枝中的特征函数与重要性定义（式 4-15、4-16）

| 论文内容 | 代码位置 | 说明 |
|---|---|---|
| 式 (4-15) \(v(S)=-E(f_{\mathcal{G}_S}(\mathbf{X}),\mathbf{Y})\) | [scorers/shapley.py](scorers/shapley.py#L202) 的 `_evaluate_loss` | 在验证集 `val_mask` 上计算交叉熵，返回 `-loss`。 |
| 式 (4-16) 边 Shapley 定义 | [scorers/shapley.py](scorers/shapley.py#L218) 的 `owen_shapley_two_stage`、[scorers/shapley.py](scorers/shapley.py#L365) 的 `shapley_approximation` 等 | 四种估计器：`owen`、`approximation`、`kernelshap`、`marginal`。 |
| 联盟值缓存 | `CoalitionValueFunction` 的 `self.cache`，键为 `(base_masks_hash, unit_type, target_layer, coalition_tuple)` | 避免重复前向；论文未展开，但实验依赖。 |
| 全局图上的 Shapley（非局部子图） | `CoalitionValueFunction` 直接用整张图的 `adj_norm` | 与 GNNShap 的“k-hop 局部子图”不同，见 2.12 节。 |

**四种估计器对照**：

| 方法名 | 函数 | 论文对应 |
|---|---|---|
| `owen` | `owen_shapley_two_stage`（第 218 行） | 第三章 Two-stage Owen 采样算法；第四章用于 Owen+Owen 实验。 |
| `approximation` | `shapley_approximation`（第 365 行） | Monte Carlo 排列采样（Approx-SV）。 |
| `kernelshap` | `shapley_kernelshap`（第 400 行） | Shapley 核加权最小二乘（KernelSHAP）。 |
| `marginal` | `shapley_marginal_sampling`（第 472 行） | 随机子集 + 随机玩家的边际贡献采样（Marginal-SV）。 |

### 2.8 4.3.1～4.3.2 toy 图构造与直观分析

论文的 toy 图：7 个节点 \(\{v_0,\dots,v_6\}\)，7 条无向边：

```text
(0,1), (0,2), (0,5), (0,6), (1,3), (2,4), (5,6)
```

标签：\(v_0,\dots,v_4\) 为类 0，\(v_5,v_6\) 为类 1。传播：两步邻域平均，式 (4-17) \(\mathbf{H}^{(1)}=\hat{\mathbf{A}}\mathbf{X},\ \mathbf{H}^{(2)}=\hat{\mathbf{A}}\mathbf{H}^{(1)}\)，论文称无参数。

| 脚本 | 图 | 传播/模型 | 评估范围 | 与论文关系 |
|---|---|---|---|---|
| [toy_model2.py](toy_model2.py#L70) | **7 节点、7 边**（正确） | `TwoStepMeanClassifier`，含 `nn.Linear(2,2)` 分类头 | 仅目标节点 \(v_0\) | 表 4-1 的“局部/目标节点”版本；ES 精确 Shapley 可用。 |
| [toy_model4.py](toy_model4.py#L46) | **7 节点、7 边**（正确） | `ToyGCNValueFunction`，含随机初始化的 `nn.Linear(2,2)` | 全部节点（`val_mask` 全 1） | 表 4-2 的五种联盟划分采样版本。 |
| [toy_model3.py](toy_model3.py#L62) | **8 节点、8 边**（含 `(3,4)`、`(6,7)`） | `TwoStepMeanClassifier`，含分类头 | 全部节点 | “全局”版本，但图比论文多一个节点/一条边。 |
| [toy_model5.py](toy_model5.py#L48) | **8 节点、8 边**（同上） | `ToyGCNValueFunction`，含随机分类头 | 全部节点 | toy_model4 的 8 节点扩展版，额外算 Exact Shapley。 |

**必须注意的差异**：

1. 论文说 toy 传播是**无参数**的（式 4-17，\(\logits=\mathbf{H}^{(2)}\)），但 `toy_model2/3/4/5` 的**当前有效代码**都保留了一个 `nn.Linear(2,2)` 分类头；`toy_model3.py` 中无参数版本被注释掉了。
2. `toy_model3.py`/`toy_model5.py` 构造的是 8 节点图，不是论文的 7 节点图。
3. `toy_model4.py`/`toy_model5.py` 没有设置 `torch.manual_seed`，且分类头随机初始化，因此每次运行的 Owen 值会变化；`toy_model2.py`/`toy_model3.py` 在 `main()` 里设置了 `torch.manual_seed(42)`。

这些差异解释了为什么下面表 4-1/4-2 只能定性复现，而不能逐位复现。

### 2.9 4.3.3 表 4-1：不同归因方法在 toy 图七条边上的全局重要性评分

表 4-1 的八列方法对应：

| 论文方法 | 代码实现 | 位置 |
|---|---|---|
| Feature-L2 | `feature_l2_score` | [scorers/edge_scorers.py](scorers/edge_scorers.py#L159) |
| Sensitivity | `sensitivity_score` | [scorers/edge_scorers.py](scorers/edge_scorers.py#L276) |
| Grad | `grad_score` | [scorers/edge_scorers.py](scorers/edge_scorers.py#L262) |
| Taylor | `taylor_score` | [scorers/edge_scorers.py](scorers/edge_scorers.py#L267) |
| Marginal-SV | `shapley_marginal_sampling` | [scorers/shapley.py](scorers/shapley.py#L472) |
| KernelSHAP | `shapley_kernelshap` | [scorers/shapley.py](scorers/shapley.py#L400) |
| Approx-SV | `shapley_approximation` | [scorers/shapley.py](scorers/shapley.py#L365) |
| Exact-SV | `toy_shapley_exact` / `compute_exact_toy_edge_shapley` / `exact_shapley_values` | [toy_model2.py](toy_model2.py#L495)、[toy_model3.py](toy_model3.py#L452)、[toy_model5.py](toy_model5.py#L164) |

**运行命令**：

```bash
cd /home/niclas/shapley/GCN_pruning

# 7 节点图 + 目标节点 v0 + exact/approx/kernelshap（对应表4-1的局部版本）
../.venv/bin/python toy_model2.py

# 8 节点“全局”版本 + 四个基线（toy_model3 的 main 目前只跑四个基线）
../.venv/bin/python toy_model3.py
```

**复现等级**：🟡

- 符号模式与论文一致：`(0,5)`、`(0,6)` 为负，`(1,3)`、`(2,4)` 为正，`(0,1)`、`(0,2)` 接近 0。
- 数值不一致：`toy_model2.py` 的 Exact-SV 为约 `(0,1)=0.0452, (0,5)=-0.1079, (1,3)=0.0181, (5,6)=0.0266`，而论文表 4-1 的 Exact-SV 为约 `(0,1)≈0, (0,5)=-0.029, (1,3)=0.004, (5,6)=0.010`。差异来自 2.8 节列出的分类头、图和评估范围不同。
- 要逐位复现表 4-1，需要拿到论文作者当时使用的**无参数 7 节点全局 value function**；仓库中没有这个组合脚本。

### 2.10 4.3.4 表 4-2：五种联盟划分下的 Owen-Shapley 值

| 论文联盟策略 | 代码策略名 | 代码类 |
|---|---|---|
| Source-Node | `source_node` | `SourceNodeEdgeBuilder` |
| Target-Node | `target_node` | `TargetNodeEdgeBuilder` |
| Community | `community`（`method='louvain'`） | `CommunityEdgeBuilder` |
| Motif-Bridge | `motif_bridge` | `MotifEdgeBuilder` |
| Motif-Triangle | `motif_triangle` | `MotifEdgeBuilder` |

**运行命令**：

```bash
cd /home/niclas/shapley/GCN_pruning

# 7 节点图，五种联盟策略，Two-stage Owen 采样
../.venv/bin/python toy_model4.py

# 8 节点图，六种策略，Two-stage Owen + Exact Shapley
../.venv/bin/python toy_model5.py
```

**复现等级**：🟡（符号模式一致，数值不一致）

- `toy_model4.py` 的输出与论文表 4-2 的**符号模式完全一致**：`(0,1)`、`(0,2)`、`(0,5)`、`(0,6)` 为正，`(1,3)`、`(2,4)`、`(5,6)` 为负。
- 数值差距较大。例如 `source_node` 下 `toy_model4` 给出的 `(0,1)=0.0055`、`(0,5)=0.0267`、`(5,6)=-0.1499`；论文表 4-2 是 `0.0378`、`0.0396`、`-0.1754`。
- **论文表 4-2 标题写“精确计算”，但表 4-2 各列的列和随联盟划分变化**（Source-Node 约 -0.274，Target-Node 约 -0.123，Community 约 -0.244，Motif-Bridge/Triangle 约 -0.032）。精确 Owen 值满足效率性 \(\sum_i \phi_i=v(N)-v(\varnothing)\)，列和应当相同；因此表 4-2 更像二阶段采样结果，或不同策略使用了不同的 value function/归一化。仓库只实现了**采样版** `owen_shapley_two_stage`，没有精确 Owen 枚举。逐位复现表 4-2 需要作者当时的 value function、随机种子和采样参数，仓库中未保留。

### 2.11 4.4.1 实验设置

论文 4.4.1 节写：Cora、约 2708 节点、5000 余条无向边、1433 维特征、7 类、140/500/1000 划分，两层 GCN，一次性剪枝。

代码中 Cora 的实际加载结果（已核对）：

```bash
cd /home/niclas/shapley/GCN_pruning
../.venv/bin/python -c "from data import load_dataset; d=load_dataset('Cora','data/'); print(d.num_nodes, d.num_features, d.num_classes, d.edge_index.size(1), int(d.train_mask.sum()), int(d.val_mask.sum()), int(d.test_mask.sum()))"
# 2708 1433 7 10556 140 500 1000
```

- 10556 是**有向**边数；无向边约 5278 条（`edge_index` 双向存储）。
- 数据加载：[data/loader.py](data/loader.py)，Planetoid 公共划分。
- 设置 A（表 4-3/4-4）的模型：`../GNNShap/dataset/configs.py` 的 `Cora` 配置，hidden 16、2 层。
- 设置 B（表 4-5/4-6）的模型：`run_large_scale_experiments.py` 的 `train_base_model()`，hidden 64、3 层 GCN + 1 层 MLP。
- **“两层 GCN”只适用于设置 A**；设置 B 是三层 GCN + MLP。这是论文文字与代码的一个不一致点。

### 2.12 4.4.2 算法 4-1/4-2、表 4-3/4-4、图 4-8～4-11

这一小节是**跨目录**最明显的地方：论文的“局部子图解释 → 聚合到全局 → 一次性剪枝”流程主要在 `GNNShap` 中完成；GCN_pruning 提供的是**全局 Shapley**和掩码语义。

#### 2.12.1 算法 4-1（节点级重要性估计与一次性节点剪枝）

| 论文步骤 | GNNShap 对应 | GCN_pruning 对应 |
|---|---|---|
| 以 \(v_i\) 为中心抽 \(L\)-hop 子图 | `GNNShapExplainer` 的 `num_hops` | ❌ 无局部子图抽取；直接在全图采样玩家 |
| 子图节点/边作为玩家 | `run_gnnshap.py`、`run_two_stage_owen.py --player_type node` | `scorers/shapley.py`，`unit_type='node'` |
| 局部 Shapley 估计 | `GNNShapSampler` + WLS/WLR | `compute_shapley_scores(...)` |
| 聚合到原图节点 | `node_shap.py`、`node_sa.py` 的聚合函数 | `experiments/run_joint_pruning.py` 中直接把玩家分数写回全局张量；**但只把验证节点选为玩家**（第 134–137 行），其余节点分数为 0 |
| 按比例构造节点掩码 | `ev_node.py` 的 `prune_graph_by_ratio*` | `JointPruningRunner.create_graph_mask`（第 346 行）+ `apply_node_mask` |

**重要差异**：论文算法 4-1 是“对所有待解释节点做局部解释，再聚合”；GCN_pruning 的 `JointPruningRunner` 是“直接在全图上对验证节点做全局 Shapley”，非验证节点没有分数。两者不是同一个算法，但共享 Shapley 估计器。复现论文表 4-3 必须走 GNNShap 的局部解释 + 聚合路线。

#### 2.12.2 算法 4-2（边级重要性估计与一次性边剪枝）

| 论文步骤 | GNNShap 对应 | GCN_pruning 对应 |
|---|---|---|
| \(L\)-hop 子图、边作为玩家 | `run_gnnshap.py`、`run_two_stage_owen.py --player_type edge` | `scorers/shapley.py`，`unit_type='edge'`，全图边玩家 |
| 局部边 Shapley | `GNNShapSampler` / `Two-stage Owen` | `owen_shapley_two_stage` / `shapley_approximation` |
| 聚合到原图边 | `edge_shap.py::aggregate_edge_shapley_to_original_graph`（第 6 行） | 直接把全局边分数写入 `edge_scores` |
| 按比例边删除 | `ev_edge.py::prune_edges_by_ratio`（第 31 行） | `apply_edge_mask` + `create_graph_mask` |

#### 2.12.3 表 4-3：节点消息屏蔽下的测试准确率

**结论：✅ 可精确复现。** 论文表 4-3 的四行分别来自 GNNShap 的四个节点分数文件；评估函数是 `../GNNShap/ev_node.py` 中的 `systematic_pruning_evaluation_node_isolation`（第 482 行，消息屏蔽/节点隔离）。

| 论文行 | 分数文件 | 生成方式 |
|---|---|---|
| PGExplainer | `../GNNShap/results/mean/node_pge_mean.pkl` | `baselines/run_pgexplainer.py` + `node_sa.py` 聚合 |
| Saliency | `../GNNShap/results/mean/node_sa_mean.pkl` | `baselines/run_sa.py` + `node_sa.py` 聚合 |
| GNNShap | `../GNNShap/results/mean/node_gnnshap_mean.pkl` | `run_gnnshap.py` + `node_shap.py` 聚合 |
| Two-stage-Shapley | `../GNNShap/results/two_stage_owen/Cora_node_greedy_modularity_CR3_CS3_K10_seed0.pkl` | `run_two_stage_owen.py --player_type node ...` |

评估脚本的实现就是论文式 (4-7)-(4-9)：

```python
# ../GNNShap/ev_node.py::prune_graph_by_ratio_node_isolation
keep = ~(iso_mask[src] | iso_mask[dst])   # 被隔离节点不参与任何消息传递
ei_new = ei[:, keep]
ei_new, _ = add_self_loops(ei_new, num_nodes=num_nodes)  # 被隔离节点保留自环
```

**精确复现命令**（不需要改任何文件）：

```bash
cd /home/niclas/shapley/GNNShap
../.venv/bin/python - <<'PY'
import json
# 只执行 ev_node.py 的函数定义部分，跳过文件末尾的 __main__ 调用
src = open('ev_node.py').read().split('if __name__ == "__main__":')[0]
exec(src)

items = [
    ('PGExplainer',        'results/mean/node_pge_mean.pkl'),
    ('Saliency',           'results/mean/node_sa_mean.pkl'),
    ('GNNShap',            'results/mean/node_gnnshap_mean.pkl'),
    ('Two-stage-Shapley',  'results/two_stage_owen/Cora_node_greedy_modularity_CR3_CS3_K10_seed0.pkl'),
]
out = {}
for name, pkl in items:
    out[name] = systematic_pruning_evaluation_node_isolation(pkl, 'Cora')

json.dump(out, open('/tmp/ch4_table43.json', 'w'), indent=2)
print('saved /tmp/ch4_table43.json')
PY
```

前 10 个稀疏度点（0.0–0.9）与论文表 4-3 完全一致；函数还会多输出 1.0 点。

#### 2.12.4 表 4-4：边删除下的测试准确率

**结论：✅ 可精确复现。** 四个结果文件与论文表 4-4 逐项一致：

| 论文行 | 结果文件 | 生成方式 |
|---|---|---|
| PGExplainer | `../GNNShap/edge_pruning_ratio_pge_results.json` | `baselines/run_pgexplainer.py` → `edge_shap.py` 聚合 → `ev_edge.py` |
| GNNShap | `../GNNShap/results/edge_pruning_ratio_results.json` | `run_gnnshap.py` → `edge_shap.py` 聚合 → `ev_edge.py` |
| Random | `../GNNShap/results/random_edge_pruning_ratio_results.json` | `random_edge.py` |
| Two-stage-Shapley | `../GNNShap/edge_mean_results.json` | `run_two_stage_owen.py --player_type edge ...` → `ev_edge.py` |

即时核对命令（无需重新跑实验）：

```bash
cd /home/niclas/shapley/GNNShap
../.venv/bin/python - <<'PY'
import json
files = {
    'PGExplainer':       'edge_pruning_ratio_pge_results.json',
    'GNNShap':           'results/edge_pruning_ratio_results.json',
    'Random':            'results/random_edge_pruning_ratio_results.json',
    'Two-stage-Shapley': 'edge_mean_results.json',
}
for name, p in files.items():
    rows = json.load(open(p))
    print(f'{name:18s}', [round(r['accuracy'], 4) for r in rows])
PY
```

输出与论文表 4-4 完全相同。

#### 2.12.5 图 4-8～4-11（三种节点语义 × 四种方法）

| 图 | 论文内容 | 代码对应 | 复现等级 |
|---|---|---|---|
| 图 4-8/4-9 | 节点删除、节点特征屏蔽、节点消息屏蔽三种语义下的性能曲线（均值/求和聚合） | `../GNNShap/ev_node.py` 的三个函数：`systematic_pruning_evaluation`（删除，第 199 行）、`systematic_pruning_evaluation_feature_mask`（特征屏蔽，第 440 行）、`systematic_pruning_evaluation_node_isolation`（消息屏蔽，第 482 行） | 🟡 方法齐全；结果 JSON 只保留了部分曲线 |
| 图 4-10/4-11 | 节点消息屏蔽 vs 边删除 | `ev_node.py`（消息屏蔽）+ `ev_edge.py`（边删除） | 🟡/🔵 数据可生成，论文的多面板图需要自己重绘 |

**“均值聚合”与“求和聚合”**：对应 GNNShap 中 `node_sa.py`/`node_shap.py` 的 `aggregation='mean'/'sum'`。现有 `results/mean/` 下是 mean 聚合的 pkl；sum 聚合需要重新运行 `node_sa.py`/`node_shap.py` 并改 `aggregation_methods`。

### 2.13 4.4.3 GCN 上的联合剪枝实验结果（算法 4-3、图 4-12、表 4-5/4-6）

#### 2.13.1 算法 4-3：基于 Shapley 准则的 GNN 联合剪枝框架

| 论文步骤 | 代码位置 |
|---|---|
| 计算边 Shapley-Owen 分数 \(\Phi_e\) | [experiments/run_joint_pruning.py](experiments/run_joint_pruning.py#L102) 的 `compute_graph_scores`（支持 `owen`/`approximation`，第 128–171 行） |
| 计算通道 Shapley-Owen 分数 \(\Phi_c\) | [experiments/run_joint_pruning.py](experiments/run_joint_pruning.py#L205) 的 `compute_model_scores`（第 256–326 行） |
| 按 \(r_g\) 保留 top-\(r_g|\mathcal{E}|\) 条边 | `create_graph_mask`（第 346 行） |
| 按 \(r_m\) 保留 top-\(r_m|C|\) 个通道 | `create_channel_masks`（第 377 行起） |
| 同时应用图掩码和通道掩码并评估 | `evaluate_joint_pruning`（第 450 行起） |
| 保存 `results.json`/`results.csv` | `run_experiment`（第 523 行起）、`_save_results`（第 628 行） |

**注意**：根目录的 [run_joint_pruning.py](run_joint_pruning.py#L477) 是另一个较旧版本，主要供 `run_dataset_benchmark.py` 使用；它不支持 `owen` 的 `coalitions` 参数。大规模实验脚本使用的是 `experiments` 包中的版本：

```python
# run_large_scale_experiments.py 第 23 行
from experiments import JointPruningRunner, create_keep_ratio_grid
```

#### 2.13.2 图 4-12：统一评分准则下的联合剪枝性能比较

**结论：✅ 数据可精确对应。** 图 4-12 的六个方法正是 `analyze_results.py` 默认绘制的六条曲线：

| 图例 | 代码方法 |
|---|---|
| `feature_l2 + activation_l2` | 边 `feature_l2` + 通道 `activation_l2` |
| `grad + grad` | 边 `grad` + 通道 `grad` |
| `owen + owen` | 边 Owen（`community_ch64`）+ 通道 Owen（64） |
| `random + random` | 边随机 + 通道随机 |
| `sensitivity + sensitivity` | 边 `sensitivity` + 通道 `sensitivity` |
| `taylor + taylor` | 边 `taylor` + 通道 `taylor` |

数据来源：`outputs/large_scale/exp1_baseline_methods/`。绘图脚本：`analyze_results.py`（默认 `INPUT_DIRS` 包含上述六个方法，其中 Owen 使用 `seed*_owen_community_ch64`）。仓库中已生成的图：

```text
outputs/large_scale/exp1_baseline_methods/_plots/A_test_acc_vs_graph_sparsity.png
```

该图与论文图 4-12 的数据完全对应（论文图是英文重绘版）。复现命令见 3.5 节。

#### 2.13.3 表 4-5：基线方法在不同 keep ratio 下的测试准确率（Mean±Std）

**结论：✅ 与旧结果文件逐项一致；其中 `degree+magnitude` 行可用检查点逐位复现。**

表 4-5 的原始结果在：

```text
outputs/large_scale_experiments/exp1_baseline_methods/all_results.json
```

对应关系：

| 论文行 | `graph_scoring_method` | `model_scoring_method` |
|---|---|---|
| degree+magnitude | `degree` | `magnitude` |
| degree+random | `degree` | `random` |
| random+magnitude | `random` | `magnitude` |
| random+random | `random` | `random` |

**重要**：当前 `run_large_scale_experiments.py --mode exp1` 已经被改写成“边方法 + Owen 变体”的实验（见第 213–532 行），**不再生成表 4-5 的 degree/random 节点方法**。表 4-5 是旧版 `experiment_1` 的产物，结果仍保存在 `outputs/large_scale_experiments/exp1_baseline_methods/`。复现表 4-5 请使用 3.3 节的检查点脚本。

#### 2.13.4 表 4-6：联合剪枝准确率网格（行：图保留率 \(r_g\)；列：模型保留率 \(r_m\)）

**结论：✅ 可精确复现。**

表 4-6 的数值与以下旧结果文件逐项一致：

```text
outputs/large_scale_experiments/exp4_joint_grid/all_results.json
```

该实验是 `degree + magnitude` 的 4×4 网格，3 个 seed（42/123/456）取平均。**它不是**当前 `outputs/large_scale/exp4_joint_grid_owen/` 中的 Owen 网格。当前 `run_large_scale_experiments.py --mode exp4` 已被改成 Owen 版本（`owen_target_node_ch32`），因此表 4-6 需要用 3.4 节的检查点脚本复现，或直接读取旧结果文件。

### 2.14 4.4.4 实验结果讨论与 4.5 本章小结

这两节是文字总结，没有独立代码。它们依赖的实验对应关系已经在 2.12–2.13 节给出：

- “边删除整体优于节点消息屏蔽” → 表 4-3 vs 表 4-4；
- “模型参数稀疏化对 GCN 的影响强于图结构稀疏化” → 表 4-6 的 4×4 网格；
- “owen+owen 在高稀疏率下更稳” → 图 4-12；
- “Shapley 能给出正负归因” → toy 表 4-1/4-2 与边 Shapley 的符号。

---

## 3. 复现所有实验的命令

### 3.1 环境与数据检查

```bash
cd /home/niclas/shapley/GCN_pruning

# 解释器与依赖
../.venv/bin/python -c "import torch, torch_geometric; print('torch', torch.__version__, 'pyg', torch_geometric.__version__)"

# Cora 加载与标准划分
../.venv/bin/python -c "from data import load_dataset; d=load_dataset('Cora','data/'); print(d.num_nodes, d.num_features, d.num_classes, d.edge_index.size(1), int(d.train_mask.sum()), int(d.val_mask.sum()), int(d.test_mask.sum()))"
# 期望：2708 1433 7 10556 140 500 1000
```

如果数据缓存缺失，PyG 会自动从 Planetoid 下载；CiteSeer/PubMed 的符号链接说明见项目根目录的 `EXPERIMENT_RESULTS.md`。第四章只使用 Cora，不需要额外处理。

### 3.2 toy 图实验（表 4-1、表 4-2）

```bash
cd /home/niclas/shapley/GCN_pruning

# 表4-1：7节点图，目标节点v0，feature_l2/sensitivity/grad/taylor + approx/kernelshap/exact
../.venv/bin/python toy_model2.py | tee /tmp/ch4_toy_model2.log

# 表4-1：8节点“全局”版本，四个基线（toy_model3 的 main 目前只跑基线）
../.venv/bin/python toy_model3.py | tee /tmp/ch4_toy_model3.log

# 表4-2：7节点图，五种联盟策略，Two-stage Owen 采样
../.venv/bin/python toy_model4.py | tee /tmp/ch4_toy_model4.log

# 表4-2：8节点图，六种策略，Two-stage Owen + Exact Shapley
../.venv/bin/python toy_model5.py | tee /tmp/ch4_toy_model5.log
```

**预期**：

- `toy_model2.py`：Exact-SV 中 `(0,5)`、`(0,6)` 为负，`(1,3)`、`(2,4)` 为正；`(0,1)`、`(0,2)` 为正但较小。
- `toy_model4.py`：五种策略均给出论文表 4-2 的符号模式；数值随随机分类头初始化变化。
- `toy_model5.py`：额外打印 Exact Shapley 列，可用于观察采样 Owen 与精确 Shapley 的差异。

**注意**：这些脚本的运行时间都在秒到分钟级；`toy_model2.py` 的 Exact-SV 需要枚举 \(2^{14}\) 个有向边联盟，约 3 秒（有缓存）。

### 3.3 表 4-5：节点/通道基线联合剪枝（检查点快速复现）

表 4-5 的基座模型检查点已经保存在：

```text
outputs/large_scale_experiments/base_models/base_model_seed{42,123,456}/checkpoints/best_model.pth
```

下面的脚本直接加载这些检查点，**不重新训练**，用 `experiments.JointPruningRunner` 重跑表 4-5 的四种方法：

```bash
cd /home/niclas/shapley/GCN_pruning
../.venv/bin/python - <<'PY'
import json, statistics, torch
from models import GCNNet
from data import load_dataset
from experiments import JointPruningRunner

data = load_dataset('Cora', 'data/')
methods = [
    ('degree', 'magnitude'),
    ('degree', 'random'),
    ('random', 'magnitude'),
    ('random', 'random'),
]
ratios = [(r, r) for r in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]]
rows = {}

for seed in [42, 123, 456]:
    ckpt = f'outputs/large_scale_experiments/base_models/base_model_seed{seed}/checkpoints/best_model.pth'
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = GCNNet(
        in_dim=data.num_features, hidden_dim=64, num_classes=data.num_classes,
        num_gcn_layers=3, num_mlp_layers=1, dropout=0.5,
    )
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    runner = JointPruningRunner(
        model=model, data=data, graph_unit='node',
        device=torch.device('cpu'), seed=seed,
    )

    for gm, mm in methods:
        res = runner.run_experiment(
            graph_scoring_method=gm,
            model_scoring_method=mm,
            keep_ratios=ratios,
            channel_strategy='global',
            output_dir=f'/tmp/ch4_table45/seed{seed}_{gm}_{mm}',
        )
        for r in res:
            key = (gm, mm, round(r['graph_keep_ratio'], 2))
            rows.setdefault(key, []).append(r['test_acc'])

print('method, keep, mean, std, n')
for (gm, mm, keep), vals in sorted(rows.items()):
    print(f'{gm}+{mm}, {keep:.2f}, {statistics.mean(vals):.4f}, '
          f'{statistics.pstdev(vals):.4f}, {len(vals)}')
PY
```

**结果说明**：

- `degree+magnitude` 行与论文表 4-5 **逐位一致**（已用 seed 42/123 验证）。
- `degree+random`、`random+magnitude`、`random+random` 行依赖随机评分实现。当前 `scorers/node_scorers.py` 的 `random_score` 用 `torch.rand`，与生成旧结果时的随机序列不同，**重跑不会逐位一致**；这三行请以 `outputs/large_scale_experiments/exp1_baseline_methods/all_results.json` 为准。
- 若只需要核对论文数值，见 4.4 节的一行 Python 命令。

### 3.4 表 4-6：4×4 联合剪枝网格（检查点快速复现）

```bash
cd /home/niclas/shapley/GCN_pruning
../.venv/bin/python - <<'PY'
import json, statistics, torch
from models import GCNNet
from data import load_dataset
from experiments import JointPruningRunner

data = load_dataset('Cora', 'data/')
grid = [(g, m) for g in [1.0, 0.8, 0.6, 0.4] for m in [1.0, 0.8, 0.6, 0.4]]
cells = {}

for seed in [42, 123, 456]:
    ckpt = f'outputs/large_scale_experiments/base_models/base_model_seed{seed}/checkpoints/best_model.pth'
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = GCNNet(
        in_dim=data.num_features, hidden_dim=64, num_classes=data.num_classes,
        num_gcn_layers=3, num_mlp_layers=1, dropout=0.5,
    )
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    runner = JointPruningRunner(
        model=model, data=data, graph_unit='node',
        device=torch.device('cpu'), seed=seed,
    )
    res = runner.run_experiment(
        graph_scoring_method='degree',
        model_scoring_method='magnitude',
        keep_ratios=grid,
        channel_strategy='global',
        output_dir=f'/tmp/ch4_table46/seed{seed}',
    )
    for r in res:
        key = (round(r['graph_keep_ratio'], 2), round(r['model_keep_ratio'], 2))
        cells.setdefault(key, []).append(r['test_acc'])

print('rg/rm, 1.00, 0.80, 0.60, 0.40')
for g in [1.0, 0.8, 0.6, 0.4]:
    row = []
    for m in [1.0, 0.8, 0.6, 0.4]:
        row.append(f"{statistics.mean(cells[(g, m)]):.4f}")
    print(f'{g:.2f}, ' + ', '.join(row))
PY
```

**结果说明**：该脚本与论文表 4-6 逐位一致（3 seeds 平均）。它使用 `degree + magnitude`，与旧结果 `outputs/large_scale_experiments/exp4_joint_grid/all_results.json` 对应。

### 3.5 图 4-12：统一评分准则下的联合剪枝曲线

图 4-12 的数据已经保存在：

```text
outputs/large_scale/exp1_baseline_methods/
```

#### 方式 A：直接用现有结果绘图（最快）

`analyze_results.py` 默认的 `ROOT` 和字体都是作者机器上的路径，需要改成本地路径：

1. 将第 28 行改为：

```python
ROOT = Path("/home/niclas/shapley/GCN_pruning/outputs/large_scale/exp1_baseline_methods")
```

2. 将第 12 行的中文字体改为仓库自带的字体（当前 venv 环境没有 `/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc`）：

```python
font_path_cn = os.path.expanduser("fonts/SIMSUNB.TTF")
```

3. 安装绘图依赖（当前 venv 未安装 `matplotlib`）：

```bash
cd /home/niclas/shapley/GCN_pruning
../.venv/bin/pip install matplotlib
```

然后运行：

```bash
cd /home/niclas/shapley/GCN_pruning
../.venv/bin/python analyze_results.py
```

输出：

```text
outputs/large_scale/exp1_baseline_methods/_plots/A_test_acc_vs_graph_sparsity.png
outputs/large_scale/exp1_baseline_methods/_plots/B_test_acc_vs_model_sparsity.png
outputs/large_scale/exp1_baseline_methods/_plots/C_leaderboard_test_acc.png
```

`A_test_acc_vs_graph_sparsity.png` 与论文图 4-12 的数据完全对应。默认 `INPUT_DIRS` 已包含六个方法，其中 `owen + owen` 使用 `seed*_owen_community_ch64`。

#### 方式 B：重新生成数据（完整重跑，耗时较长）

```bash
cd /home/niclas/shapley/GCN_pruning
../.venv/bin/python run_large_scale_experiments.py --mode exp1
```

- 输出目录：`outputs/large_scale/exp1_baseline_methods/`
- 该脚本会为每个 seed 重新训练基座模型，并跑 6 个基线组合 + 6 个 Owen 变体 × 6 个 keep ratio。
- 默认 seeds 在 `run_large_scale_experiments.py` 第 61 行：`[42, 123, 456]`。为了快速冒烟，可以临时改成 `[42]`。
- 如果只想画 Owen 变体，运行 `analyze2.py`（同样需要把 `ROOT` 改成本地路径，并安装 `matplotlib`），输出在 `_plots_owen/`。

### 3.6 表 4-3：节点消息屏蔽（GNNShap）

```bash
cd /home/niclas/shapley/GNNShap
../.venv/bin/python - <<'PY'
import json
src = open('ev_node.py').read().split('if __name__ == "__main__":')[0]
exec(src)

items = [
    ('PGExplainer',       'results/mean/node_pge_mean.pkl'),
    ('Saliency',          'results/mean/node_sa_mean.pkl'),
    ('GNNShap',           'results/mean/node_gnnshap_mean.pkl'),
    ('Two-stage-Shapley', 'results/two_stage_owen/Cora_node_greedy_modularity_CR3_CS3_K10_seed0.pkl'),
]
out = {}
for name, pkl in items:
    out[name] = systematic_pruning_evaluation_node_isolation(pkl, 'Cora')
json.dump(out, open('/tmp/ch4_table43.json', 'w'), indent=2)
print('saved /tmp/ch4_table43.json')
PY
```

该命令直接复现论文表 4-3 的前 10 个点（0.0–0.9）。

**如果要重新生成四个分数文件**：

```bash
cd /home/niclas/shapley/GNNShap

# PGExplainer
../.venv/bin/python baselines/run_pgexplainer.py --dataset Cora --repeat 1

# Saliency（需要 captum）
../.venv/bin/python baselines/run_sa.py --dataset Cora --repeat 5

# GNNShap（需要 colorlog；GNNShapSampler 还可能需要编译 CUDA 扩展）
../.venv/bin/python run_gnnshap.py --dataset Cora --num_samples 10000 --repeat 1 \
    --batch_size 1024 --sampler GNNShapSampler --solver WLSSolver

# Two-stage Owen（节点）
../.venv/bin/python run_two_stage_owen.py --dataset Cora --player_type node \
    --coalition_mode greedy_modularity --CR 3 --CS 3 --K 10 --seed 0 --device cpu
```

聚合脚本：

- `node_sa.py`：聚合 SA/PGExplainer 的节点分数（第 6 行 `aggregate_node_sa_scores`）。
- `node_shap.py`：把边解释聚合为节点分数（第 27 行 `aggregate_edge_to_node_scores_complete`）。

这些脚本的 `pkl_path` 也是作者机器上的 `/root/Desktop/...`，需要改成本地路径。

### 3.7 表 4-4：边删除（GNNShap）

#### 方式 A：直接用现有结果核对（最快）

```bash
cd /home/niclas/shapley/GNNShap
../.venv/bin/python - <<'PY'
import json
files = {
    'PGExplainer':       'edge_pruning_ratio_pge_results.json',
    'GNNShap':           'results/edge_pruning_ratio_results.json',
    'Random':            'results/random_edge_pruning_ratio_results.json',
    'Two-stage-Shapley': 'edge_mean_results.json',
}
for name, p in files.items():
    rows = json.load(open(p))
    print(f'{name:18s}', [round(r['accuracy'], 4) for r in rows])
PY
```

#### 方式 B：重新生成

```bash
cd /home/niclas/shapley/GNNShap

# 1) PGExplainer 解释
../.venv/bin/python baselines/run_pgexplainer.py --dataset Cora --repeat 1

# 2) GNNShap 解释（需要 colorlog；如无 CUDA 扩展可改用 SVXSampler）
../.venv/bin/python run_gnnshap.py --dataset Cora --num_samples 10000 --repeat 1 \
    --batch_size 1024 --sampler GNNShapSampler --solver WLSSolver

# 3) Two-stage Owen 边分数
../.venv/bin/python run_two_stage_owen.py --dataset Cora --player_type edge \
    --coalition_mode greedy_modularity --CR 3 --CS 3 --K 10 --seed 0 --device cpu

# 4) 随机边基线
../.venv/bin/python random_edge.py

# 5) 聚合解释分数到原图边：编辑 edge_shap.py 的 pkl_path 和输出名后运行
../.venv/bin/python edge_shap.py

# 6) 边删除评估：编辑 ev_edge.py 末尾的 edge_shapley_pkl 为本地 pkl 后运行
../.venv/bin/python ev_edge.py
```

**注意**：

- `edge_shap.py` 的默认 `pkl_path` 指向 `Cora_PGExplainer_run1.pkl`，输出 `original_edge_pge_mean.pkl`；对 GNNShap 需要改成 `results/Cora_GNNShap_...pkl` 并改输出名。
- `ev_edge.py` 默认路径是作者机器的 `/root/Desktop/我的网盘/...`；改成上一步生成的聚合 pkl 或 `results/two_stage_owen/Cora_edge_greedy_modularity_CR3_CS3_K10_seed0.pkl`。
- `Two-stage Owen` 的 pkl 已经直接包含 `edge_importance` 字段，可以被 `ev_edge.py` 直接读取。

### 3.8 当前代码的 Owen 4×4 网格（补充实验）

```bash
cd /home/niclas/shapley/GCN_pruning
../.venv/bin/python run_large_scale_experiments.py --mode exp4
```

- 输出：`outputs/large_scale/exp4_joint_grid_owen/joint_matrix_test_acc.csv`
- 配置：边 Owen `target_node` + 通道 Owen `gcn_layers.0`，`max_edge_players=100`，`channel_player_cap=32`，`num_samples_q=5`、`num_samples_p=5`、`K=10`，3 seeds，4×4 网格。
- **这不是论文表 4-6**。论文表 4-6 是 `degree + magnitude` 的旧网格；这个 Owen 网格对应 4.4.3 节对 Owen 组合的讨论，可用于补充分析。

### 3.9 一键命令汇总

```bash
# ========== 0. 环境 ==========
cd /home/niclas/shapley/GCN_pruning
../.venv/bin/python -c "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__)"

# ========== 1. 掩码/前向 sanity ==========
../.venv/bin/python sanity_forward.py
../.venv/bin/python sanity_graph_mask.py
../.venv/bin/python sanity_model_mask.py

# ========== 2. toy 图（表4-1/4-2） ==========
../.venv/bin/python toy_model2.py
../.venv/bin/python toy_model3.py
../.venv/bin/python toy_model4.py
../.venv/bin/python toy_model5.py

# ========== 3. 表4-5 / 表4-6（用检查点，见3.3/3.4的完整脚本） ==========
# 直接读取旧结果核对：
../.venv/bin/python - <<'PY'
import json, statistics
d = json.load(open('outputs/large_scale_experiments/exp1_baseline_methods/all_results.json'))
for gm in ['degree', 'random']:
    for mm in ['magnitude', 'random']:
        vals = [r['test_acc'] for r in d
                if r['graph_scoring_method'] == gm and r['model_scoring_method'] == mm]
        print(gm, mm, 'n=', len(vals), 'mean=', round(statistics.mean(vals), 4))
PY

# ========== 4. 图4-12（现有结果直接绘图） ==========
# 先安装 matplotlib；把 analyze_results.py 第28行 ROOT 改成：
# /home/niclas/shapley/GCN_pruning/outputs/large_scale/exp1_baseline_methods
# 再把第12行字体改成 "fonts/SIMSUNB.TTF"
../.venv/bin/pip install matplotlib
../.venv/bin/python analyze_results.py

# ========== 5. 表4-3（GNNShap，见3.6完整脚本） ==========
cd /home/niclas/shapley/GNNShap
# 运行3.6节的 Python heredoc

# ========== 6. 表4-4（GNNShap，见3.7） ==========
../.venv/bin/python - <<'PY'
import json
files = {
    'PGExplainer': 'edge_pruning_ratio_pge_results.json',
    'GNNShap': 'results/edge_pruning_ratio_results.json',
    'Random': 'results/random_edge_pruning_ratio_results.json',
    'Two-stage-Shapley': 'edge_mean_results.json',
}
for name, p in files.items():
    print(name, [round(r['accuracy'], 4) for r in json.load(open(p))])
PY
```

---

## 4. 结果文件索引与数值核对

### 4.1 文件索引

| 论文元素 | 结果文件/输出 |
|---|---|
| 表 4-1 | `toy_model2.py` / `toy_model3.py` 的控制台输出（未自动保存；可用 `tee`） |
| 表 4-2 | `toy_model4.py` / `toy_model5.py` 的控制台输出 |
| 表 4-3 | `../GNNShap/results/mean/node_{pge,sa,gnnshap}_mean.pkl` + `../GNNShap/ev_node.py::systematic_pruning_evaluation_node_isolation` |
| 表 4-4 | `../GNNShap/edge_pruning_ratio_pge_results.json`、`results/edge_pruning_ratio_results.json`、`results/random_edge_pruning_ratio_results.json`、`edge_mean_results.json` |
| 表 4-5 | `outputs/large_scale_experiments/exp1_baseline_methods/all_results.json` |
| 表 4-6 | `outputs/large_scale_experiments/exp4_joint_grid/all_results.json` |
| 图 4-8～4-11 | `../GNNShap/ev_node.py`、`ev_edge.py` 生成；现有部分 JSON 未覆盖全部语义/聚合 |
| 图 4-12 | `outputs/large_scale/exp1_baseline_methods/_plots/A_test_acc_vs_graph_sparsity.png` |
| Owen 4×4 补充网格 | `outputs/large_scale/exp4_joint_grid_owen/joint_matrix_test_acc.csv` |

### 4.2 表 4-3 核对

论文表 4-3（0.0–0.9）的四行：

```text
PGExplainer        0.8150 0.8000 0.7920 0.7820 0.7730 0.7710 0.7600 0.7520 0.7310 0.7200
Saliency           0.8150 0.8080 0.7950 0.7770 0.7510 0.7530 0.7460 0.7340 0.7190 0.7080
GNNShap            0.8150 0.7980 0.7540 0.7330 0.7330 0.7240 0.7200 0.7190 0.7150 0.7120
Two-stage-Shapley  0.8150 0.7840 0.7590 0.7330 0.7230 0.7200 0.7230 0.7180 0.7100 0.7100
```

3.6 节的命令会输出完全相同的数值（第 11 个点 1.0 也会输出，论文未列）。

### 4.3 表 4-4 核对

```text
PGExplainer        0.8150 0.8100 0.8010 0.7930 0.7910 0.7860 0.7850 0.7790 0.7690 0.7490
GNNShap            0.8150 0.8090 0.8090 0.8040 0.8060 0.8060 0.7930 0.7880 0.7700 0.7530
Random             0.8150 0.8080 0.8010 0.7970 0.7930 0.7900 0.7800 0.7730 0.7500 0.7420
Two-stage-Shapley  0.8150 0.8120 0.8150 0.7990 0.7830 0.7710 0.7640 0.7670 0.7520 0.7260
```

### 4.4 表 4-5 核对

从 `outputs/large_scale_experiments/exp1_baseline_methods/all_results.json` 计算得到（3 seeds，Mean±Std）：

| Method | 1.00 | 0.90 | 0.80 | 0.70 | 0.60 | 0.50 |
|---|---:|---:|---:|---:|---:|---:|
| degree+magnitude | 0.7707±0.0076 | 0.7580±0.0120 | 0.7277±0.0257 | 0.6443±0.0228 | 0.5290±0.0679 | 0.3950±0.0274 |
| degree+random | 0.7707±0.0076 | 0.7587±0.0151 | 0.7260±0.0128 | 0.6417±0.0348 | 0.4830±0.0545 | 0.3663±0.0555 |
| random+magnitude | 0.7707±0.0076 | 0.7423±0.0213 | 0.7137±0.0267 | 0.6367±0.0252 | 0.5187±0.0580 | 0.3913±0.0160 |
| random+random | 0.7707±0.0076 | 0.7493±0.0207 | 0.7120±0.0211 | 0.6810±0.0135 | 0.6203±0.0210 | 0.4500±0.0643 |

与论文表 4-5 完全一致。

### 4.5 表 4-6 核对

从 `outputs/large_scale_experiments/exp4_joint_grid/all_results.json` 计算得到（3 seeds 平均）：

| \(r_g\backslash r_m\) | 1.00 | 0.80 | 0.60 | 0.40 |
|---|---:|---:|---:|---:|
| 1.00 | 0.7707 | 0.7453 | 0.5373 | 0.2213 |
| 0.80 | 0.7480 | 0.7277 | 0.5380 | 0.2260 |
| 0.60 | 0.7303 | 0.7087 | 0.5290 | 0.2303 |
| 0.40 | 0.6957 | 0.6750 | 0.5140 | 0.2340 |

与论文表 4-6 完全一致。

---

## 5. 已知差异与复现边界

### 5.1 表 4-1/4-2：只能定性复现

| 原因 | 影响 |
|---|---|
| 论文式 (4-17) 是**无参数**两步传播；`toy_model2/3/4/5` 当前有效代码都带 `nn.Linear(2,2)` 分类头 | 数值尺度不同，符号模式一致 |
| `toy_model3.py`/`toy_model5.py` 使用 **8 节点、8 边**图；论文是 **7 节点、7 边** | 边集合不同，无法逐列对齐 |
| `toy_model4.py`/`toy_model5.py` 没有固定随机种子，分类头随机初始化 | 每次运行 Owen 值不同 |
| 表 4-2 标称“精确计算”，但列和随联盟划分变化，违反 Owen 效率性 | 论文表 4-2 可能实际是采样结果；仓库没有精确 Owen 枚举 |
| 论文未完整给出 toy 特征值、归一化细节、验证节点集合 | 即使改成无参数传播，也无法唯一确定论文数值 |

**能做到**：公式 (4-7)-(4-16) 的实现、五种联盟划分、正负号模式、Exact Shapley 与采样 Owen 的对比。
**做不到**：逐位复现表 4-1/4-2 的数值。

### 5.2 表 4-3/4-4：可精确复现，但依赖 GNNShap 的预训练模型和结果文件

- 表 4-3/4-4 的基线 0.815 来自 GNNShap 的 **2 层 hidden-16 GCN**，不是 GCN_pruning 的 3 层 hidden-64 模型。
- 表 4-3 的评估语义是**节点消息屏蔽/隔离**（`ev_node.py::systematic_pruning_evaluation_node_isolation`），不是节点删除。若误用 `systematic_pruning_evaluation`（删除），会得到不同的曲线。
- 表 4-4 的评估是**边删除**（`ev_edge.py::prune_edges_by_ratio`），只在“可解释边”集合内按比例删除，未解释的边始终保留。
- 跨目录依赖：`captum`（Saliency）、`colorlog`（GNNShapSampler）、`python-louvain`/`pymetis`（部分联盟构建）当前 venv 未安装；需要时安装：

```bash
cd /home/niclas/shapley
.venv/bin/pip install captum colorlog python-louvain
```

- GNNShap 的 CUDA 扩展 `cppextension/cudagnnshap.cu` 尚未编译；`GNNShapSampler` 可能需要首次运行时编译，或改用 `SVXSampler`。

### 5.3 表 4-5/4-6：结果可精确核对，重跑脚本已变

- 表 4-5/4-6 的结果是**旧版** `experiment_1`/`experiment_4` 的产物，保存在 `outputs/large_scale_experiments/`。
- 当前 `run_large_scale_experiments.py` 的 `--mode exp1`/`--mode exp4` 已被改写为**边方法 + Owen** 版本，不再生成表 4-5/4-6。
- 用 3.3/3.4 节的检查点脚本可以复现：
  - 表 4-6：完全一致；
  - 表 4-5：`degree+magnitude` 行完全一致；三个随机行依赖随机评分实现，重跑不保证逐位一致。
- 表 4-6 是 `degree+magnitude` 网格，**不是** `outputs/large_scale/exp4_joint_grid_owen/` 的 Owen 网格。

### 5.4 图 4-12：数据一致，绘图脚本需改路径

- 数据在 `outputs/large_scale/exp1_baseline_methods/`，六条曲线与论文图 4-12 一致。
- `analyze_results.py` 的 `ROOT`、`INPUT_DIRS` 和字体路径都是作者机器上的绝对路径（`/root/Desktop/我的网盘/...`、`/usr/share/fonts/...`），必须改成本地路径；当前 venv 还需要安装 `matplotlib`。
- `owen+owen` 曲线默认使用 `seed*_owen_community_ch64`；如果换用 `target_node_ch32` 等变体，曲线会不同。论文图 4-12 对应的是 `community_ch64` 这一组（与默认 `INPUT_DIRS` 一致）。

### 5.5 代码级注意事项

1. **根目录 `run_joint_pruning.py` vs `experiments/run_joint_pruning.py`**：
   - 根目录版本主要给 `run_dataset_benchmark.py` 使用，`compute_graph_scores` 对 `owen` 没有转发 `players/coalitions`，直接调用会报错。
   - 大规模实验使用 `experiments` 包版本，支持 `players/coalitions/num_samples_q/num_samples_p/K/target_layer`。
2. **节点 Shapley 的玩家选择**：`experiments/run_joint_pruning.py` 的 `JointPruningRunner.compute_graph_scores` 只从 `val_mask` 选节点（第 134–137 行），其余节点分数为 0；论文算法 4-1 是对所有解释节点做局部解释再聚合。两者不可混用。
3. **边 Shapley 的玩家空间**：GCN_pruning 的玩家是 `edge_index` 的列索引（有向边），Cora 有 10556 个；GNNShap 的局部解释是 k-hop 子图内的边，聚合后是无向边（5278 条）。表 4-4 的“可解释边 5278”对应后者。
4. **剪枝是软掩码**：GCN_pruning 的通道剪枝是 forward 中乘 0/1 掩码，不是 TorchPruner 那种真正删除参数张量的硬剪枝；论文也采用掩码语义。
5. **当前 exp1/exp4 的 seed**：`run_large_scale_experiments.py` 第 61 行硬编码 `[42, 123, 456]`；没有 CLI 参数。快速冒烟时临时改成 `[42]`。
6. **`analyze_results.py`/`analyze2.py`/`vis4.py` 的路径**：这三个脚本都包含作者机器的 `/root/Desktop/我的网盘/...` 绝对路径，运行前必须改 `ROOT`/`INPUT_JSON`。

---

## 6. 代码阅读路线与关键行号

### 6.1 建议阅读顺序

1. [models/gcn_layer.py](models/gcn_layer.py#L10)：`GCNConv`、`normalize_adjacency`；
2. [models/gcn_net.py](models/gcn_net.py#L11)：`GCNNet`、`list_prunable_layers`；
3. [pruning/graph_mask.py](pruning/graph_mask.py#L68)：`apply_edge_mask`、`apply_node_mask`；
4. [pruning/model_mask.py](pruning/model_mask.py#L10)：`MaskedModelWrapper`；
5. [scorers/shapley.py](scorers/shapley.py#L53)：`CoalitionValueFunction`、四种估计器；
6. [coalition_builder/edge_builders.py](coalition_builder/edge_builders.py#L520)：边联盟；
7. [experiments/run_joint_pruning.py](experiments/run_joint_pruning.py#L29)：`JointPruningRunner`；
8. [run_large_scale_experiments.py](run_large_scale_experiments.py#L213)：实验 1/3/4；
9. [toy_model2.py](toy_model2.py#L70) / [toy_model4.py](toy_model4.py#L46)：7 节点 toy 图；
10. `../GNNShap/ev_node.py` / `ev_edge.py`：论文表 4-3/4-4 的评估。

### 6.2 关键行号速查

| 文件 | 行号 | 内容 |
|---|---:|---|
| `models/gcn_layer.py` | 46 | `GCNConv.forward` |
| `models/gcn_layer.py` | 75 | `normalize_adjacency` |
| `models/gcn_net.py` | 86 | `GCNNet.forward` |
| `models/gcn_net.py` | 132 | `list_prunable_layers` |
| `pruning/graph_mask.py` | 68 | `apply_edge_mask` |
| `pruning/graph_mask.py` | 104 | `apply_node_mask`（消息屏蔽） |
| `pruning/model_mask.py` | 10 | `MaskedModelWrapper` |
| `scorers/shapley.py` | 53 | `CoalitionValueFunction` |
| `scorers/shapley.py` | 148 | `_evaluate_edge_coalition` |
| `scorers/shapley.py` | 202 | `_evaluate_loss` |
| `scorers/shapley.py` | 218 | `owen_shapley_two_stage` |
| `scorers/shapley.py` | 365 | `shapley_approximation` |
| `scorers/shapley.py` | 400 | `shapley_kernelshap` |
| `scorers/shapley.py` | 472 | `shapley_marginal_sampling` |
| `scorers/edge_scorers.py` | 159 | `feature_l2_score` |
| `scorers/edge_scorers.py` | 262 | `grad_score` |
| `scorers/edge_scorers.py` | 267 | `taylor_score` |
| `scorers/edge_scorers.py` | 276 | `sensitivity_score` |
| `coalition_builder/edge_builders.py` | 52 | `SourceNodeEdgeBuilder` |
| `coalition_builder/edge_builders.py` | 90 | `TargetNodeEdgeBuilder` |
| `coalition_builder/edge_builders.py` | 127 | `CommunityEdgeBuilder` |
| `coalition_builder/edge_builders.py` | 263 | `MotifEdgeBuilder` |
| `coalition_builder/edge_builders.py` | 520 | `build_edge_coalitions` |
| `experiments/run_joint_pruning.py` | 102 | `compute_graph_scores` |
| `experiments/run_joint_pruning.py` | 205 | `compute_model_scores` |
| `experiments/run_joint_pruning.py` | 523 | `run_experiment` |
| `run_large_scale_experiments.py` | 213 | `experiment_1_baseline_methods` |
| `run_large_scale_experiments.py` | 593 | `experiment_4_joint_pruning_grid` |
| `toy_model2.py` | 70 | 7 节点 toy 图 |
| `toy_model2.py` | 495 | `toy_shapley_exact` |
| `toy_model3.py` | 452 | `compute_exact_toy_edge_shapley` |
| `toy_model4.py` | 46 | 7 节点 toy 图（Owen） |
| `toy_model4.py` | 278 | `run_strategy` |
| `toy_model5.py` | 164 | `exact_shapley_values` |
| `../GNNShap/ev_node.py` | 199 | 节点删除评估 |
| `../GNNShap/ev_node.py` | 364 | `prune_graph_by_ratio_node_isolation` |
| `../GNNShap/ev_node.py` | 440 | 节点特征屏蔽评估 |
| `../GNNShap/ev_node.py` | 482 | 节点消息屏蔽评估 |
| `../GNNShap/ev_edge.py` | 31 | `prune_edges_by_ratio` |
| `../GNNShap/ev_edge.py` | 97 | `edge_pruning_evaluation_ratio` |
| `../GNNShap/edge_shap.py` | 6 | 边解释聚合 |
| `../GNNShap/node_sa.py` | 6 | SA/PGE 节点聚合 |
| `../GNNShap/node_shap.py` | 27 | 边解释→节点聚合 |
| `../GNNShap/run_two_stage_owen.py` | 27 | Two-stage Owen CLI |
| `../GNNShap/baselines/run_pgexplainer.py` | 38 | PGExplainer |
| `../GNNShap/baselines/run_sa.py` | 42 | Saliency |
| `../GNNShap/dataset/configs.py` | 19 | Cora 2 层 hidden-16 配置 |

---

## 附录 A：第四章目录（论文原文）

```text
第4章 基于 Shapley 值的 GNN 剪枝研究 ................ 38
4.1 GNN 图结构剪枝的特点与挑战 ..................... 39
    4.1.1 图结构剪枝中的剪枝单元定义 ............... 39
    4.1.2 节点剪枝与边剪枝对消息传递的影响 ......... 41
    4.1.3 图结构剪枝对象的选择分析 ................. 43
4.2 GNN 图结构剪枝的合作博弈建模 ................... 44
    4.2.1 边作为玩家的合作博弈表示 ................. 44
    4.2.2 图结构中的联盟划分方式 ................... 44
    4.2.3 图结构剪枝中的特征函数与重要性定义 ....... 45
4.3 Shapley 值用于图结构剪枝的理论合理性 ........... 46
    4.3.1 Toy 图构造 ............................... 46
    4.3.2 简单图结构的剪枝直观分析 ................. 47
    4.3.3 Shapley 理想属性与理想图剪枝要求的对应 ... 48
    4.3.4 图结构联盟划分与 Two-stage Owen 的适配性 . 49
4.4 GNN 剪枝实验结果与分析 ......................... 50
    4.4.1 实验设置 ................................. 50
    4.4.2 边剪枝与点剪枝的比较 ..................... 50
    4.4.3 GCN 上的联合剪枝实验结果 ................. 53
    4.4.4 实验结果讨论 ............................. 57
4.5 本章小结 ....................................... 58
```

## 附录 B：论文元素与复现等级总表

| 论文元素 | 代码/数据 | 等级 |
|---|---|---|
| 式 (4-1)(4-2)、图 4-1 | `models/gcn_layer.py`、`models/gcn_net.py` | ✅ |
| 式 (4-3)(4-4) 节点删除 | `../GNNShap/ev_node.py::systematic_pruning_evaluation` | 🔵 |
| 式 (4-5)(4-6) 节点特征屏蔽 | `../GNNShap/ev_node.py::systematic_pruning_evaluation_feature_mask` | 🔵 |
| 式 (4-7)～(4-9) 节点消息屏蔽 | `pruning/graph_mask.py::apply_node_mask` | ✅ |
| 式 (4-10)～(4-13) 边掩码 | `pruning/graph_mask.py::apply_edge_mask` | ✅ |
| 图 4-2～4-6 | 无绘图脚本 | ❌ |
| 命题 4.1～4.3 | 掩码函数 + toy 符号结果 | 🟡 |
| 边作为玩家 | `scorers/shapley.py::CoalitionValueFunction` | ✅ |
| 边联盟划分 | `coalition_builder/edge_builders.py` | ✅ |
| 式 (4-15)(4-16) | `CoalitionValueFunction` + 四种估计器 | ✅ |
| 7 节点 toy 图 | `toy_model2.py` / `toy_model4.py` | ✅（图构造） |
| 表 4-1 | `toy_model2/3` + `scorers/edge_scorers.py` | 🟡 |
| 表 4-2 | `toy_model4/5` + `coalition_builder` | 🟡 |
| 算法 4-1（局部节点） | `../GNNShap` 的 `run_gnnshap.py`/`run_two_stage_owen.py` + `node_shap.py` | 🔵 |
| 算法 4-2（局部边） | `../GNNShap` 的 `run_gnnshap.py`/`run_two_stage_owen.py` + `edge_shap.py` | 🔵 |
| 表 4-3 | `../GNNShap` pkls + `ev_node.py` 消息屏蔽 | ✅ |
| 表 4-4 | `../GNNShap` 四个结果 JSON | ✅ |
| 图 4-8～4-11 | `../GNNShap/ev_node.py`/`ev_edge.py` | 🟡 |
| 算法 4-3 | `experiments/run_joint_pruning.py::JointPruningRunner` | ✅ |
| 图 4-12 | `outputs/large_scale/exp1_baseline_methods/` + `analyze_results.py` | ✅（数据） |
| 表 4-5 | `outputs/large_scale_experiments/exp1_baseline_methods/all_results.json` | ✅（旧结果）/ 🟡（重跑随机行） |
| 表 4-6 | `outputs/large_scale_experiments/exp4_joint_grid/all_results.json` | ✅ |
| 4.4.4 讨论、4.5 小结 | 文字 | — |

---

## 附录 C：跨目录依赖安装与常见问题

### C.1 可选依赖

```bash
cd /home/niclas/shapley
# 绘图（analyze_results.py / analyze2.py）
.venv/bin/pip install matplotlib
# Saliency 基线
.venv/bin/pip install captum
# GNNShapSampler 的日志依赖和社区发现
.venv/bin/pip install colorlog python-louvain
```

- `analyze_results.py` 还依赖中文字体路径；当前环境没有 `wqy-zenhei.ttc`，请把第 12 行改成仓库自带的 `fonts/SIMSUNB.TTF`。
- `matplotlib` 未安装时，`analyze_results.py`/`analyze2.py` 会直接报 `ModuleNotFoundError`；仓库中已有的 `_plots/*.png` 可以直接查看，不需要重新绘图。

- `pymetis` 在当前 Python 3.12 上可能难以安装；GNNShap 的 `greedy_modularity` 不依赖它，`louvain` 可用 `python-louvain`。
- GNNShap 的 CUDA 扩展未编译时，`run_gnnshap.py` 可尝试 `--sampler SVXSampler`。

### C.2 常见错误

| 错误 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: No module named 'scorers'` | 对 `toy_model*.py` 使用了 `-I` | 去掉 `-I`，在 `GCN_pruning` 目录下运行 |
| `ValueError: Owen method requires 'coalitions' argument` | 使用了根目录 `run_joint_pruning.py` 的旧版 runner（`run_dataset_benchmark.py --prune shapley` 也会走到这里） | 改用 `experiments.JointPruningRunner`；当前 `run_dataset_benchmark.py` 的 Owen 路径需要先修正 runner 的 `coalitions` 转发 |
| `ModuleNotFoundError: No module named 'captum'` | 未安装 Saliency 依赖 | `pip install captum` |
| `ModuleNotFoundError: No module named 'colorlog'` | GNNShap 依赖缺失 | `pip install colorlog` |
| `analyze_results.py` 读不到文件 | `ROOT` 还是作者机器路径 | 改成本地 `outputs/...` 路径 |
| 表 4-5 的 random 行与论文不一致 | 随机评分实现/RNG 状态不同 | 以旧 `all_results.json` 为准；随机基线不要求逐位复现 |
| toy 表 4-1/4-2 数值与论文不一致 | 分类头、8 节点图、随机种子差异 | 接受定性一致；逐位复现需作者原始脚本 |

---

**文档结束。**
