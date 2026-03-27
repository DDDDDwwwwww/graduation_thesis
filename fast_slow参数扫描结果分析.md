# 参数扫描结果分析
目的：解决初步实验结果中slow分支始终未被触发的问题。

# stage1 粗扫
- 在stage1中运行命令：
nohup python experiments4two_stage/run_experiment_ts_g_gate_sweep.py \
  --device cuda \
  --games games/breakthrough.kif \
  --rounds 8 \
  --taus 0.001 0.003 0.005 0.01 0.02 \
  --visit-thresholds 0 1 2 \
  --slow-budgets 4 8 \
  > outputs4two_stage/logs/ts_g_reduced_stage1.log 2>&1 &
- 结果分析：
从outputs4two_stage\experiments\TS_G_gate_sweep结果看：

mean_mean_slow_call_ratio_a = 0
mean_mean_slow_trigger_rate_a = 0
mean_mean_uncertainty_ok_rate_a = 0
也就是说，**这一轮里 slow 还是一次都没触发。**

更关键的是，不确定度分布已经把问题定位出来了：

mean_mean_u_fast_p95_a 大约都在**0.000535 ~ 0.000541**
mean_mean_u_fast_mean_a 大约都在**0.000323 ~ 0.000327**
而你这轮扫的**最小 tau 是：0.001**
这说明：**你的 tau 仍然比 u_fast_p95 大了接近一倍**
所以 uncertainty_ok 基本必然为假
**slow 不触发的主因已经不是 visit_threshold，而是 tau 量级 still too high**

这轮 stage1 已经足够告诉我们：

visit 信号修复是有必要的，但现在真正卡住 slow 的主要是 tau
下一轮应该把 tau 继续往下压到 1e-4 量级
visit_threshold 不需要大扫了，先保留 0/1 就够
保留的方向：
先保留：
visit_threshold = 0 和 1
slow_budget_per_move = 4 和 8

先不优先保留：
visit_threshold = 2
这轮没有显示出额外价值
而且会进一步增加 slow 触发难度
一句话总结：现在最该调的是 tau，而且 tau 要下探到 0.0001 ~ 0.0006，不是 0.001+

# Stage2 参数扫描结果分析
- 实验输出：outputs4two_stage\experiments\TS_G_gate_sweep_stage2_tau_low
- 运行命令：
nohup python experiments4two_stage/run_experiment_ts_g_gate_sweep.py \
  --device cuda \
  --games games/breakthrough.kif \
  --rounds 10 \
  --taus 0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 \
  --visit-thresholds 0 1 \
  --slow-budgets 4 8 \
  --out-dir outputs4two_stage/experiments/TS_G_gate_sweep_stage2_tau_low \
  > outputs4two_stage/logs/ts_g_stage2_tau_low.log 2>&1 &
## 1. 实验目的与设置
为解决前一轮（outputs4two_stage\experiments\TS_G_gate_sweep）参数扫描中 slow 分支始终未被触发的问题，本轮实验在 `breakthrough` 上进一步下调门控阈值 `tau`，并围绕更可能触发 slow 分支的参数区间进行扫描。实验固定 `gate_type=combined`、`uncertainty_type=variance_head`，仅比较以下参数组合：

- `tau ∈ {0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006}`
- `visit_threshold ∈ {0, 1}`
- `slow_budget_per_move ∈ {4, 8}`
- 对手包括 `neural_mcts:token_transformer` 与 `pure_mct`
- 每组参数在 `breakthrough` 上进行 10 局对弈

本轮实验的核心目标不是直接追求最高胜率，而是确认 slow 分支能否稳定触发，并据此找到兼顾性能与开销的候选参数。

## 2. 核心结果
本轮实验最重要的结果是：**slow 分支已经被成功触发**。与上一轮 `TS_G_gate_sweep` 中 `mean_slow_call_ratio_a=0` 的情况相比，本轮多组参数均出现了非零 slow 调用比例，说明前面定位的问题是正确的，即原始 `tau` 设定明显高于 fast 模型的实际不确定度量级。

从不确定度分布看，`u_fast` 的高分位数大致稳定在 `5.3e-4 ~ 5.4e-4` 左右，而本轮扫描的有效 `tau` 区间集中在 `1e-4 ~ 5e-4`。这说明将 `tau` 下探到 `1e-4` 量级后，门控机制才真正进入可工作区间。

从 slow 触发比例看，表现最明显的组合主要集中在：

- `visit_threshold=0`
- `slow_budget_per_move=8`
- `tau=0.0001 ~ 0.0005`

其中，对 `neural_mcts:token_transformer` 而言：
- `tau=0.0001, visit_threshold=0, slow_budget=8` 时，`slow_call_ratio≈0.0745`，胜率达到 `1.0`，平均决策时间约为 `0.357s/步`。
- `tau=0.0003, visit_threshold=0, slow_budget=8` 时，`slow_call_ratio≈0.0758`，胜率为 `0.7`。
- `tau=0.0005, visit_threshold=0, slow_budget=8` 时，`slow_call_ratio≈0.0891`，但胜率下降到 `0.3`。

这说明：**更高的 slow 触发率并不必然带来更好的对局表现**。当 `tau` 继续增大时，虽然 slow 分支依然能够频繁触发，但整体胜率反而下降，说明过于激进的 slow 调用会破坏当前搜索中的性能平衡。

## 3. 参数影响分析
### 3.1 `tau` 的影响
`tau` 是本轮最关键的控制变量。实验结果表明：

- 当 `tau=0.0006` 时，slow 分支再次完全不触发，说明该阈值已高于当前 fast 模型的不确定度有效区间。
- 当 `tau` 位于 `0.0001 ~ 0.0004` 时，slow 分支可稳定触发。
- 当 `tau` 增大到 `0.0005` 时，虽然部分组合的 `slow_call_ratio` 达到本轮最高，但胜率明显下降。

因此，`tau` 不能仅通过“让 slow 多触发”来确定，更合理的做法是将其设在一个既能触发 slow、又不会显著损害对局性能的区间。当前结果表明，这一区间更接近 `0.0001 ~ 0.0003`，而不是更大的 `0.0004 ~ 0.0005`。

### 3.2 `visit_threshold` 的影响
比较 `visit_threshold=0` 与 `1` 可以发现：

- `visit_threshold=0` 时，slow 调用比例整体更高，触发更稳定。
- `visit_threshold=1` 时，slow 仍可触发，但触发率整体下降，且多数情况下没有带来更稳定的胜率提升。

这说明在当前实现与任务设置下，`visit_threshold=0` 更适合作为默认配置。也就是说，一旦 fast 模型给出足够高的不确定度，就应允许 slow 分支立即介入，而不必额外等待访问次数积累。

### 3.3 `slow_budget_per_move` 的影响
比较 `slow_budget_per_move=4` 与 `8` 可以发现：

- `budget=8` 通常能将 `slow_call_ratio` 提升到约 `0.05 ~ 0.08`，更容易达到预设的合理触发区间。
- `budget=4` 也能触发 slow，但触发率多落在 `0.02 ~ 0.04`，相对偏低。

因此，如果目标是验证双阶段机制确实参与了搜索过程，那么 `slow_budget_per_move=8` 更合适；如果后续目标转向进一步压缩开销，则可以再把 `budget=4` 作为低成本备选方案。

## 4. 可保留的候选参数
综合 slow 触发率、对 `neural_mcts:token_transformer` 的胜率以及平均决策时间，当前最值得保留的参数组合是：

### 主候选参数
- `tau=0.0001`
- `visit_threshold=0`
- `slow_budget_per_move=8`

选择理由：
- slow 分支已稳定触发，`slow_call_ratio≈0.0745`
- 胜率达到本轮最佳水平
- 平均决策时间仍保持在约 `0.357s/步`，没有出现明显失控

### 次候选参数
- `tau=0.0003`
- `visit_threshold=0`
- `slow_budget_per_move=8`

选择理由：
- slow 分支同样稳定触发，`slow_call_ratio≈0.0758`
- 胜率略低于主候选，但仍具备进一步验证价值
- 可作为更保守门控设定的对照组

### 低开销备选参数
- `tau=0.0001`
- `visit_threshold=0`
- `slow_budget_per_move=4`

选择理由：
- slow 调用比例约为 `0.036`
- 适合作为“较低 slow 使用率”条件下的成本对照组

## 5. 结论与下一步工作
本轮 `stage2` 参数扫描表明，双阶段门控机制已经不再停留在“fast-only”状态，而是能够在合适阈值下真实触发 slow 分支。实验进一步说明：当前 fast 模型的不确定度量级约在 `1e-4 ~ 1e-3` 内，因此门控阈值 `tau` 必须下调到 `1e-4` 量级才有实际作用。与此同时，`visit_threshold=0` 与 `slow_budget_per_move=8` 的组合最有利于稳定激活 slow 分支，并在性能与开销之间取得相对更好的平衡。

基于本轮结果，后续实验不再需要继续做更细粒度的大规模参数搜索，而应进入“候选参数回测”阶段。具体而言，可优先保留以下两组配置：

- 主配置：`tau=0.0001, visit_threshold=0, slow_budget_per_move=8`
- 对照配置：`tau=0.0003, visit_threshold=0, slow_budget_per_move=8`

随后使用这两组参数回跑主基准实验 `TS_A_main_benchmark` 和开销分析实验 `TS_F_overhead_analysis`，验证 slow 真正参与后，双阶段方法是否仍能保持相对 transformer-MCTS 的速度优势，并在跨游戏层面取得更稳定的收益。
