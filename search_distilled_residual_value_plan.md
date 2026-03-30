# 最终版执行方案：Search-Distilled Residual Phase-Aware Value Net

## 1. 方案定位

本方案用于替代当前“继续堆叠更复杂网络结构”的优化路线。核心判断是：当前实验迟迟无法超过 baseline，主要瓶颈很可能**不在网络表达能力不足**，而在于：

1. **训练标签过粗**：只用最终胜负或弱搜索回报时，监督信号噪声大。
2. **MCTS 接入方式不合理**：神经网络一旦接入全树，可能拖慢搜索，抵消价值评估收益。
3. **优化目标不够贴近 baseline 超越需求**：直接从零学习 value，难度大；但如果改为“修正 baseline 的系统误差”，更有机会稳定提升。

因此，本方案不再将主要创新点放在“新 backbone”上，而是放在：

- **更强的监督信号**：用强搜索生成 teacher value 标签；
- **更容易成功的学习目标**：残差价值学习；
- **更适合 GGP / MCTS 场景的接入方式**：选择性神经评估；
- **轻量但有效的状态增强**：phase-aware 全局特征。

---

## 2. 最终方法总览

本文最终采用的模型命名为：

**Search-Distilled Residual Phase-Aware Value Net（SDRPV-Net）**

核心思想如下：

对于状态 $s$，不再让网络直接从零预测价值 $v(s)$，而是先构造一个便宜的 baseline 估值 $b(s)$，再令网络学习对其的修正量：

$$
\hat v(s) = b(s) + \Delta_\theta(s).
$$

其中：

- $b(s)$：cheap baseline value，可来自浅层 MCTS、少量 rollout、已有 baseline 网络，三者取其一；
- $\Delta_\theta(s)$：神经网络学习的残差修正；
- teacher 标签来自更强预算的 UCT / MCTS：

$$
q_T(s) \in [-1,1].
$$

训练时，主目标不是最终胜负，而是拟合更强搜索给出的 teacher value：

$$
\Delta_\theta(s) \approx q_T(s) - b(s).
$$

推理时，网络不会无差别接入所有 MCTS 节点，而是只在关键节点调用，以减少网络推理开销对搜索预算的侵蚀。

---

## 3. 为什么这条路线比继续改网络结构更可行

### 3.1 当前失败并不说明整个项目失败

Fast-Slow Value Net 和 Token Transformer Fusion Value Net 都没能超过 baseline，这更可能说明：

- 你现在的训练数据和目标不足以支撑更复杂模型；
- 模型能力虽增强，但搜索使用方式没有让增强的能力真正转化为胜率；
- value-only 路线下，**target 质量**和**integration 设计**比 backbone 微调更重要。

因此，继续发明新 backbone 的成功概率不高。

### 3.2 残差学习比“从零学 value”更容易超过 baseline

你的真正目标不是得到最优价值函数，而是**稳定超过已有 baseline**。那么与其让网络直接学：

$$
v_\theta(s),
$$

不如令其学习：

$$
\Delta_\theta(s)=q_T(s)-b(s),
$$

因为 baseline 已经捕捉了一部分规律，网络只需补偿其系统误差，任务难度更低，也更适合论文中的“在已有强 baseline 上做有效改进”。

### 3.3 teacher search 标签比最终胜负标签更贴近搜索需求

最终胜负标签虽然自然，但存在两个问题：

1. 延迟长：中间局面到终局相隔很远；
2. 噪声大：同一局面在弱对局中得到的结果可能不稳定。

而用更强 UCT 对中间局面打出的 teacher value，本质上是在教网络拟合“更强搜索会怎么看这个局面”，这和实际在 MCTS 中调用 value 的需求是一致的。

### 3.4 选择性接入比全树接入更适合当前场景

如果你的环境中 Monte-Carlo playout 已经很快，神经网络推理反而可能成为瓶颈。此时全树都调用网络，未必比纯 MCTS 更好。正确的做法是：

- 只在真正值得调用的节点上使用网络；
- 其余节点保留原始搜索流程。

这样更有机会在固定 simulations 预算和固定时间预算下都体现优势。

---

## 4. 整体系统结构

系统由四个部分组成：

1. **状态编码器**：使用你现有的 token transformer 输入方案；
2. **cheap baseline evaluator**：生成 $b(s)$；
3. **teacher search label generator**：生成 $q_T(s)$；
4. **residual value network**：学习 $q_T(s)-b(s)$；
5. **selective MCTS integration**：在线搜索时按条件调用网络。

整体流程：

1. 用 UCT-vs-UCT 或 baseline-vs-baseline 生成对局；
2. 从对局中抽取中间状态；
3. 对每个状态计算 cheap baseline 值 $b(s)$；
4. 用更高预算 teacher UCT 计算 $q_T(s)$；
5. 训练网络拟合残差；
6. 在 MCTS 中选择性调用网络作为 leaf evaluation 修正器；
7. 与原 baseline 在同预算下比较胜率与效率。

---

## 5. 状态表示设计

### 5.1 主体表示

继续使用你当前已有的 **Board Token / Token Transformer** 输入表示，不推翻原有编码框架。

每个 tile / cell 对应一个 token，至少包含：

- tile content id
- tile position / sinusoidal position encoding
- 当前玩家信息（可作为全局特征或 special token）
- 合法性 / mask（若已有）

### 5.2 新增 phase-aware 全局特征

在 pooled representation 或 [CLS] token 后拼接以下轻量全局特征：

1. **move progress**：当前步数 / 估计最大步数
2. **board occupancy ratio**：已占用格子比例
3. **piece count difference**：双方棋子数差值（若可提取）
4. **legal move count**：当前合法动作数
5. **terminal proximity proxy**：例如无子可走、接近连线完成、接近满盘等简单指标（能提取则加，不能提取可省略）

这些特征不要求复杂规则知识，只要能从当前状态直接统计即可。

### 5.3 为什么加 phase-aware 特征

同一个局面表征在 opening / midgame / endgame 的作用不同。加入阶段性特征可以帮助模型区分：

- 开局更偏全局布局；
- 中局更偏局部交换与形势判断；
- 残局更偏精确价值。

这比再额外堆一个复杂分支网络更轻、更稳，也更容易在论文中解释。

---

## 6. cheap baseline value 的构造

本方案需要一个低成本估值 $b(s)$。优先级如下：

### 方案 A（首选）

**浅层 UCT / 小预算 MCTS 值**

例如对每个状态只跑：

- 32 simulations
- 64 simulations
- 或固定极小 time limit

然后取根节点平均回报作为：

$$
b(s).
$$

优点：

- 与最终使用场景一致；
- 不依赖额外网络；
- 可以自然作为“弱搜索基线”。

### 方案 B

**已有 baseline value net 的输出**

若你现有 baseline 网络比较稳定，也可直接取：

$$
b(s) = v_{\text{base}}(s).
$$

优点：

- 最能体现“在 baseline 上做修正”；
- 更适合作为论文的最终主线。

### 方案 C（备选）

**少量 rollout 的平均值**

若浅层 MCTS 实现麻烦，则用极少量 rollout 近似也可。

### 建议

若你的代码里浅层 MCTS 最容易实现，则优先选 **方案 A**。若你已有 baseline 网络调用方便，则并行做 **A/B 两版比较**，最后保留效果更好的一个。

---

## 7. teacher 标签构造

### 7.1 teacher 的定义

对每个被采样状态 $s$，运行一个比 student 强得多的搜索器，得到：

$$
q_T(s) \in [-1,1].
$$

teacher 可定义为：

- 强预算 UCT 根节点平均回报；或
- 强预算 UCT 根节点最优子节点对应值；或
- 多次强预算 UCT 结果的平均。

### 7.2 teacher 预算建议

若 student 在线搜索预算为 $N$ simulations，则 teacher 可设为：

- $5N$
- $10N$

例如：

- student：600 sims
- teacher：3000 或 6000 sims

### 7.3 为什么 teacher 要离线生成

teacher 只用于训练，不参与在线对局。离线生成的优点：

- 不影响对局速度；
- 可以提前缓存标签；
- 允许对一小批关键状态使用更大预算。

### 7.4 标签平滑

为避免 teacher 噪声过大，建议将回报进行平滑裁剪：

$$
q_T^{\text{clip}}(s)=\mathrm{clip}(q_T(s),-1,1).
$$

若 teacher 波动仍大，可对同一状态重复搜索 2–3 次取平均。

---

## 8. 数据集构造

### 8.1 不再只保存整局结果

训练数据以**中间状态样本**为单位，而不是仅以整局终局 winner 作为监督。

每条样本包含：

$$
(s,\; b(s),\; q_T(s),\; z(s),\; \phi_{\text{phase}}(s))
$$

其中：

- $s$：状态编码输入
- $b(s)$：cheap baseline 值
- $q_T(s)$：teacher 值
- $z(s)$：最终胜负标签（可选辅助监督）
- $\phi_{\text{phase}}(s)$：阶段全局特征

### 8.2 状态采样策略

每局不是保存所有状态，而是进行分层采样：

- opening：采样 1–2 个状态
- midgame：采样 2–4 个状态
- endgame：采样 1–2 个状态

这样能避免大量高度相似状态占据数据集。

### 8.3 困难状态优先采样

对更有区分度的局面提高采样概率，例如：

1. $|b(s)|$ 接近 0 的状态（弱搜索不确定）
2. teacher 与 baseline 差异大的状态
3. 访问次数高、位于关键分支的状态
4. 接近终局但仍有策略分歧的状态

这一步非常重要，因为它会显著提高训练样本的信息密度。

### 8.4 数据来源建议

初始版本可使用：

- baseline-vs-baseline 对局
- UCT-vs-UCT 对局
- baseline-vs-UCT 对局

混合构造数据集。这样比只用单一来源更稳。

### 8.5 可复用的已有实验资产（直接复用，减少重跑）

你当前仓库里已经有可直接复用的数据与模型，不需要从零再采一遍：

1. **现成状态数据集（可直接作为 $s,z(s)$ 初始监督）**
- `outputs/experiments/D_dataset_size_sensitivity/datasets/dataset_200.jsonl`
- `outputs/experiments/D_dataset_size_sensitivity/datasets/dataset_500.jsonl`
- `outputs/experiments/D_dataset_size_sensitivity/datasets/dataset_1000.jsonl`
- `outputs/experiments/D_dataset_size_sensitivity/datasets/dataset_2000.jsonl`

这些文件已包含状态与终局价值标签字段（如 `state_facts`, `acting_role`, `value_target`），可先直接用于 teacher-only 预热训练，再对其中采样子集做 teacher 重标注生成 $q_T(s)$ 与 $b(s)$。

2. **现成模型检查点（可用于 warm start 和 baseline evaluator）**
- `outputs/experiments/D_dataset_size_sensitivity/models/size_2000/token_transformer/model.pt`
- `outputs/experiments/D_dataset_size_sensitivity/models/size_2000/fact_mlp/model.pt`
- 以及对应 `config.json`、`encoder.json`、`vocab.json`

复用方式：
- 作为 student 初始化权重（减少收敛时间）；
- 直接作为方案 6 的 **方案 B**：令 $b(s)=v_{\text{base}}(s)$。

3. **现成评测基线与配置（可直接沿用，保证公平对照）**
- `outputs/experiments/A_baseline_strength/summary/*.json`
- `outputs/experiments/B_time_budget_sensitivity/summary/*.json`
- `outputs/experiments/C_search_budget_sensitivity/summary/*.json`
- `outputs/experiments/E_encoder_model_ablation/summary/*.json`
- `outputs/experiments/H_multi_game_benchmark/summary/*.json`
- 各实验 `meta/run_manifest.json`

这些结果可直接作为论文对照基线与门槛线，避免重复跑 A/B/C/E/H 全量实验。

### 8.6 数据格式转换脚本需求清单（旧 `dataset_*.jsonl` -> 新 `(s,b,q_T,z,\phi)`）

目标：将 `outputs/experiments/D_dataset_size_sensitivity/datasets/dataset_*.jsonl` 转为新方案训练样本格式，并支持后续增量补标注。

1. 脚本与入口
- 建议脚本路径：`experiments/convert_dataset_to_sdrpv.py`
- 支持命令行参数：
  - `--input`：单个或多个旧 jsonl 文件
  - `--output`：输出 jsonl 文件路径
  - `--game`：可选，按游戏过滤（默认不过滤）
  - `--baseline-mode`：`shallow_mcts` / `base_net`
  - `--teacher-sims`：teacher 搜索预算
  - `--student-sims`：用于设定 teacher/student 比例（可选）
  - `--resume`：断点续跑（读取已有输出并跳过已完成样本）
  - `--max-samples`：调试模式下限制处理条数

2. 输入字段要求（旧数据）
- 必须读取：`state_facts`, `acting_role`, `value_target`
- 可选读取：`game_name`, `ply_index`, `terminal`, `match_id`, `source_agent`
- 若缺失关键字段，记录错误并跳过，不中断全流程。

3. 输出字段定义（新样本）
- `s`：可直接喂给编码器的状态表示（至少保留 `state_facts` 与 `acting_role`）
- `b`：cheap baseline 值（来自浅层 MCTS 或 baseline 网络）
- `q_t`：teacher 搜索值（强预算 UCT/MCTS）
- `z`：终局监督值（由旧 `value_target` 映射，范围统一到 `[-1,1]`）
- `phi`：phase-aware 特征字典，至少包含：
  - `move_progress`
  - `board_occupancy_ratio`
  - `piece_count_diff`（可提取则填，提取失败可置空）
  - `legal_move_count`
  - `terminal_proximity_proxy`（可用简单规则近似）
- 建议附加元数据：`game_name`, `match_id`, `ply_index`, `source_agent`, `baseline_mode`, `teacher_sims`

4. 字段映射与数值规范
- `z = clip(value_target, -1, 1)`
- `b`、`q_t`、`z` 全部强制 `[-1,1]` 裁剪
- 缺失或异常数值（NaN/Inf）样本直接丢弃并计数到日志。

5. 计算与缓存要求
- baseline 与 teacher 必须支持磁盘缓存（按状态哈希键控）
- 推荐缓存文件：
  - `outputs/cache/sdrpv_baseline_cache.jsonl`
  - `outputs/cache/sdrpv_teacher_cache.jsonl`
- 同一状态重复出现时优先读缓存，避免重复搜索。

6. 采样与增量策略
- 支持 `--phase-sampling`（opening/midgame/endgame 分层采样）可选开关
- 默认先“全量转换 + 不重算已有缓存”
- 后续只对新增状态做增量转换与补标注。

7. 质量校验与验收标准
- 输出前打印并保存统计：
  - 总输入数、成功转换数、跳过数、失败数
  - `b/q_t/z` 的均值、方差、分位数
  - `|q_t-b|` 分布（用于判断残差学习是否有信息量）
- 验收最低标准：
  - 成功转换率 >= 95%
  - 无 NaN/Inf
  - `q_t` 非常数（标准差大于 0）
  - 抽样 100 条人工检查字段齐全且语义正确。

8. 与训练管线对接要求
- 输出文件命名建议：`outputs/datasets/sdrpv_dataset_v1.jsonl`
- 训练侧 dataloader 直接读取 `(s,b,q_t,z,phi)`，不再二次转换
- 支持后续切分导出：
  - `train.jsonl`
  - `val.jsonl`
  - `test.jsonl`

---

## 9. 网络结构设计

### 9.1 主体结构

保留你当前最稳定的 token transformer 主体，不再新造大网络。

建议形式：

1. token embedding
2. positional encoding
3. transformer encoder blocks
4. mean pooling 或 [CLS] pooling
5. 拼接 phase-aware global features
6. 两层 MLP head
7. 输出 residual $\Delta_\theta(s)$

### 9.2 推荐输出头

输出两个量：

1. **residual value**：$\Delta_\theta(s)$
2. **confidence / uncertainty score**：$u_\theta(s) \in [0,1]$

其中最终值：

$$
\hat v(s)=b(s)+\Delta_\theta(s)
$$

然后将其裁剪到 $[-1,1]$。

### 9.3 uncertainty 分支作用

uncertainty 用于 MCTS 接入阶段控制网络权重。例如：

- 若模型很自信，则更依赖网络；
- 若模型不自信，则更依赖 baseline / rollout。

如果实现压力太大，第一版可以先不做 uncertainty 分支，只做固定权重融合。

---

## 10. 训练目标设计

### 10.1 主目标：teacher 蒸馏

主损失设为：

$$
\mathcal L_{\text{distill}} = \mathrm{Huber}(\hat v(s), q_T(s)).
$$

使用 Huber loss 而不是纯 MSE，原因是对 teacher 噪声更稳。

### 10.2 辅助目标：最终胜负监督

同时可保留终局标签监督：

$$
\mathcal L_{\text{terminal}} = \mathrm{MSE}(\hat v(s), z(s)).
$$

其中 $z(s) \in \{-1,1\}$ 或归一化结果值。

### 10.3 残差正则

由于残差学习不希望网络输出过大修正，可加入：

$$
\mathcal L_{\text{res}} = \|\Delta_\theta(s)\|_2^2.
$$

### 10.4 总损失

最终训练目标为：

$$
\mathcal L = \lambda_1 \mathcal L_{\text{distill}} + \lambda_2 \mathcal L_{\text{terminal}} + \lambda_3 \mathcal L_{\text{res}}.
$$

推荐初始取值：

- $\lambda_1 = 1.0$
- $\lambda_2 = 0.2$
- $\lambda_3 = 10^{-4}$

如发现 teacher 标签足够稳定，可将 $\lambda_2$ 再调小。

### 10.5 若使用 uncertainty 分支

可进一步做加权蒸馏：

$$
\mathcal L_{\text{distill}} = w(s)\,\mathrm{Huber}(\hat v(s), q_T(s)),
$$

其中 $w(s)$ 可由 teacher 稳定性或样本难度决定。第一版不强制实现。

---

## 11. MCTS 接入设计

## 11.1 目标

目标不是让网络替代整个搜索，而是让网络在**关键时刻**提供更好的叶节点评估。

## 11.2 选择性调用条件

建议只在满足以下任一条件时调用网络：

1. 节点深度达到阈值 $d \ge d_{\min}$
2. 节点访问次数达到阈值 $n(s) \ge n_{\min}$
3. 节点位于根节点下一层或二层的关键分支
4. rollout / baseline 估值不确定，即 $|b(s)|$ 较小

推荐从最简单版本开始：

- 只对 **深度 >= 2 或 3** 的叶节点调用网络；
- 或只对 **根节点扩展出的子树** 调用网络。

## 11.3 融合方式

### 固定权重融合（第一版推荐）

$$
V_{\text{use}}(s) = (1-\alpha) b(s) + \alpha \hat v(s),
$$

初始可设：

- $\alpha = 0.5$
- 或 $\alpha = 0.7$

### 条件权重融合（第二版）

令：

$$
\alpha(s) = f(n(s), d(s), u_\theta(s)).
$$

例如：

- 节点访问次数越多，$\alpha(s)$ 越大；
- 模型越自信，$\alpha(s)$ 越大；
- 若模型不自信，则退回 baseline。

## 11.4 为什么不用“全树全调用”

原因是：

1. 计算开销太大；
2. 在 playout 极快的环境中，神经评估可能拖慢总体搜索；
3. 网络优势往往只在少量关键节点上最明显。

因此，本方案强调 **selective neural evaluation**，这是方法的重要组成部分。

---

## 12. 训练与推理的完整伪代码

## 12.1 数据生成阶段

```text
for each self-generated game or baseline-vs-baseline game:
    play game using UCT / baseline agent
    store trajectory
    sample states from opening / midgame / endgame
    for each sampled state s:
        compute cheap baseline value b(s)
        run stronger teacher search to obtain q_T(s)
        record final outcome z(s)
        extract phase-aware features phi(s)
        save (s, b(s), q_T(s), z(s), phi(s))
```

## 12.2 训练阶段

```text
for each minibatch:
    encode states using token transformer
    concatenate phase-aware global features
    predict residual Delta_theta(s)
    compute v_hat(s) = clip(b(s) + Delta_theta(s), -1, 1)
    compute L_distill = Huber(v_hat(s), q_T(s))
    compute L_terminal = MSE(v_hat(s), z(s))
    compute L_res = ||Delta_theta(s)||^2
    optimize L = lambda1*L_distill + lambda2*L_terminal + lambda3*L_res
```

## 12.3 在线 MCTS 阶段

```text
for each leaf node s in MCTS:
    compute cheap baseline value b(s)
    if selective_condition(s) == True:
        predict v_hat(s) using neural residual model
        use V_use(s) = (1-alpha)*b(s) + alpha*v_hat(s)
    else:
        use V_use(s) = b(s)
    backpropagate V_use(s)
```

---

## 13. 实验设计

## 13.1 总原则

本方案的实验目标不是证明“新模型在所有游戏上全面碾压”，而是证明：

1. 在固定预算下，提出的方法能**稳定优于原 baseline**；或
2. 在达到相同强度时，提出的方法需要**更少 simulations / 更少数据 / 更少训练时间**。

只要有其中一项稳定成立，论文就成立。

## 13.2 第一阶段：最小可行验证（强烈建议先做）

只做 4 个实验：

### Exp-1 原 baseline

- 当前已有 best baseline
- 原训练标签
- 原 MCTS 接入

### Exp-2 teacher distillation only

- 网络结构不变
- 只把监督改成强搜索 teacher label
- 不做残差学习

### Exp-3 teacher distillation + residual learning

- 网络学习 $q_T(s)-b(s)$
- 观察是否已经超过 baseline

### Exp-4 full method

- residual + phase-aware features + selective MCTS integration

### 第一阶段判定标准

只要 Exp-3 或 Exp-4 中任意一个稳定优于 Exp-1，就进入论文主实验。

---

## 13.3 第二阶段：正式主实验

### 比较对象

1. 纯 UCT / pure MCTS baseline
2. 原始 value-only baseline
3. 本文最终方案 SDRPV-Net

### 比较设置

#### 设置 A：固定 simulations

例如：

- 200 sims
- 600 sims
- 1000 sims

比较胜率。

#### 设置 B：固定时间预算

例如：

- 0.1s / move
- 0.5s / move
- 1s / move

比较胜率。

这个设置非常重要，因为它可以检验 selective integration 是否真的降低了神经推理带来的效率损失。

#### 设置 C：少数据训练

例如：

- 200 局训练数据
- 500 局训练数据
- 1000 局训练数据
- 2000 局训练数据

比较不同方法在小样本下的表现。

### 成功标准

至少满足以下之一：

- 固定 simulations 下，较 baseline 平均提升 3%–8% 且稳定；
- 固定时间下，优于原始全神经接入版；
- 少数据下，达到相同胜率所需数据量减少。

---

## 13.4 消融实验

为保证论文完整性，建议做如下消融：

### Ablation-1 去掉 teacher distillation

检验 teacher label 是否是主要增益来源。

### Ablation-2 去掉 residual learning

让网络直接预测 value，检验残差结构的必要性。

### Ablation-3 去掉 phase-aware global features

检验轻量全局特征是否有效。

### Ablation-4 去掉 selective integration

改成全树调用网络，检验选择性调用的重要性。

### Ablation-5 更换 cheap baseline

比较：

- 浅层 MCTS 值
- baseline 网络值
- 少量 rollout 值

确定哪一种更适合作为 $b(s)$。

---

## 14. 游戏选择建议

### 14.1 先不要全游戏铺开

当前阶段最重要的是拿到 positive result，不是覆盖所有游戏。因此先选：

- **2 个最接近出提升的游戏**作为主实验入口；
- 再选 **1 个更难游戏** 作为泛化补充。

### 14.2 游戏筛选原则

优先选择：

1. 中等规模棋盘
2. 搜索深度与分支度适中
3. 当前 baseline 已表现出一定神经信号，但还不稳定
4. 训练与评估成本可控

避免优先选择：

1. 极大棋盘
2. 胜率高度不稳定的游戏
3. 推理开销远大于 playout 收益的游戏

### 14.3 论文中的叙述方式

先在“成功率高”的游戏上验证方法机制，再将方法测试到更难游戏。这样比一上来全游戏失败更有说服力。

---

## 15. 实施顺序（必须严格按此顺序）

## 第 1 步：冻结 backbone

不要再改 token transformer 主体结构。先把 backbone 固定下来，避免变量太多。

## 第 2 步：实现中间状态采样数据集

将现有对局生成逻辑修改为：

- 保存 opening / midgame / endgame 的 sampled states
- 支持对每个状态单独打 teacher 标签

优先复用 `outputs/experiments/D_dataset_size_sensitivity/datasets/dataset_*.jsonl` 作为初始状态池，再增量补采样，不从零开始重建全量数据。

## 第 3 步：实现 cheap baseline value

优先实现浅层 MCTS 版 $b(s)$。如果已有 baseline 网络调用更方便，则同步留接口。

## 第 4 步：实现 teacher search label pipeline

给每个样本状态增加：

- stronger UCT search
- teacher value cache

务必将 teacher 标签缓存到磁盘，避免重复计算。

## 第 5 步：先做“teacher only”实验

不要一上来就做 full method。先做：

- 不改网络结构
- 只改训练目标为 teacher value

如果这一版都没有提升，再检查 teacher 预算和样本采样策略。

## 第 6 步：加入 residual learning

在已有 teacher 标签基础上改为学习：

$$
\Delta_\theta(s)=q_T(s)-b(s).
$$

这一步最关键，也是最有可能第一次超过 baseline 的地方。

## 第 7 步：加入 selective integration

训练稳定后，再接入 MCTS。不要训练和接入同时大改。

## 第 8 步：最后再加入 phase-aware features

如果前面已经有提升，phase-aware 特征用来进一步稳定和增强；如果前面还没提升，也不要把问题归咎于这些小特征。

---

## 16. 风险与应对

### 风险 1：teacher 搜索成本过高

**应对：**

- 只对采样状态打 teacher，不对所有状态打；
- 优先给困难状态打 teacher；
- teacher 预算先用 5× 而不是 10×。

### 风险 2：teacher 标签仍然太噪

**应对：**

- 对同一状态重复搜索两次后取平均；
- 改用 Huber loss；
- 对波动过大样本降权。

### 风险 3：网络接入后对局变慢，固定时间下反而变差

**应对：**

- 必须做 selective integration；
- 记录每步平均推理调用次数；
- 必要时限制每回合最多调用网络若干次。

### 风险 4：残差学习没有带来提升

**应对：**

- 更换 $b(s)$ 的来源；
- 检查 baseline 是否过弱或过强；
- 检查 teacher 与 baseline 的差值分布，若大多数样本残差接近 0，则说明 teacher 预算不够强。

### 风险 5：实验变量太多，结果混乱

**应对：**

- 严格按“teacher only → residual → selective integration → phase-aware”顺序递增；
- 每次只改一个模块；
- 保留所有中间版本结果，负结果也可进入论文附录或消融。

---

## 17. 论文中的创新点表述建议

最终论文不要把创新点写成“又提出了一个更复杂网络”。建议写成以下三点：

### 创新点 1：Search-distilled supervision for value-only GGP

针对 value-only GGP 中最终胜负监督过于粗糙的问题，提出使用更强预算搜索生成 teacher value，对中间状态进行蒸馏监督。

### 创新点 2：Residual value correction over a cheap baseline estimator

提出让网络学习对 cheap baseline 估值的修正量，而非从零直接拟合价值函数，从而更适合在已有强 baseline 上做可观测提升。

### 创新点 3：Selective neural integration into MCTS

提出只在关键节点调用神经评估的接入方式，以降低神经推理对搜索预算的侵蚀，适应 GGP 中 playout 快于推理的现实场景。

如果你后续 phase-aware 特征效果明显，也可作为补充创新点：

### 创新点 4（可选）：Phase-aware global feature augmentation

通过轻量全局特征增强状态阶段信息，提高同一网络对不同对局阶段的适应能力。

---

## 18. 论文结果不理想时的保底写法

即使最终提升不大，这个方案仍然能写出一篇完整论文，因为你可以得到以下有价值结论：

1. 单纯增加 backbone 复杂度不一定带来 value-only GGP 提升；
2. teacher supervision 比终局监督更有利于学习可用于搜索的 value；
3. 神经网络接入方式对最终胜率影响不亚于网络结构本身；
4. 在 fast playout 环境中，fixed-time evaluation 与 fixed-simulation evaluation 可能给出不同结论。

这四条中任意两条成立，论文就已经有研究价值。

---

## 19. 最终建议：你现在不要做什么

1. 不要推翻整个项目重开；
2. 不要再额外发明第三种复杂 backbone；
3. 不要一口气同时改数据、模型、搜索三部分；
4. 不要以“必须全面碾压 baseline”为唯一成功标准。

你现在真正需要的是：

- 先拿到一个稳定的 positive result；
- 再围绕这个 result 扩展消融与分析；
- 最后把负结果转化为论文中的设计动机。

---

## 20. 可直接执行的本周任务清单

### Day 1–2

- 冻结当前最稳的 token transformer backbone
- 先接入并清洗 `D_dataset_size_sensitivity` 的现有 `dataset_*.jsonl`
- 实现中间状态采样（仅做增量补充）
- 为每个状态保存 metadata（步数、棋子数、合法动作数等）

### Day 3–4

- 实现 cheap baseline value $b(s)$
- 实现 teacher UCT 标签生成与缓存
- 构建新的训练样本格式

### Day 5

- 跑 teacher-only 版本
- 与原 baseline 先做离线验证（value 预测误差 / rank correlation）

### Day 6

- 跑 residual 版本
- 首次接入 MCTS 做小规模对局测试

### Day 7

- 加 selective integration
- 记录 fixed simulations / fixed time 两组结果
- 选出最有希望的游戏进入主实验

---

## 21. 最终一句话结论

本方案的核心不是“再造一个更复杂的网络”，而是：

> **用更强搜索提供更好的监督，用残差学习降低超过 baseline 的难度，用选择性接入保证神经价值真正转化为搜索优势。**

这是一条比继续改 Fast-Slow 或 Fusion 更稳、更容易做出实验结果、也更适合写论文的最终路线。
