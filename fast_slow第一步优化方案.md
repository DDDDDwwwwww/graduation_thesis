# 双阶段门控优化计划

## Summary
- 目前看，问题不只是“阈值太高”，而是有两个原因叠加：
- `combined` 门控要求同时满足 `uncertainty_ok && visit_ok && budget_ok`，而代码里传入的是新扩展叶节点的 `leaf_node.visits`，评估发生在回传之前，所以这个值大概率一直是 `0`。这会让所有依赖 `visit_threshold` 的门控基本无法触发。
- 即使切到 `uncertainty` 门控，实验里 `slow_calls` 仍为 `0`，说明 `variance_head` 的输出量级也很可能普遍低于 `tau=0.15`，存在不确定度未校准或阈值量纲不匹配的问题。
- 结论：当前 slow 模式未触发，**更像是“门控设计 + 阈值校准”同时有问题**，不能只靠调一个 `tau` 解决。

## Key Changes
- 第一优先级先修正门控输入信号，而不是直接重训模型。
- 将 `visit_threshold` 的判定对象从“当前新叶子节点访问次数”改为更有意义的统计量，默认选：
  - 当前被展开动作的访问次数，或
  - 当前父节点访问次数。
- 不再把 `leaf_node.visits` 作为 slow 门控依据，因为在当前搜索流程里它在评估时几乎恒为 0。
- 在 `TwoStageValueEvaluator` 和实验汇总中新增门控诊断统计：
  - `uncertainty_ok_rate`
  - `visit_ok_rate`
  - `budget_ok_rate`
  - `slow_trigger_rate`
  - `u_fast` 的均值、P50、P90、P95、最大值
- 用这些统计先确认“是谁把 slow 锁死了”，再做阈值搜索。

## Optimization Approach
- 第一步：做一次门控可观测性验证。
  - 在 `TS_A` 或单独 smoke 实验里打印每步 `u_fast` 分布和三类 gate 条件命中率。
  - 验证 `visit_ok_rate` 是否接近 0；如果是，先修 visit 信号。
- 第二步：修完 visit 信号后，做小规模阈值扫描。
  - `tau` 默认扫：`0.001 / 0.005 / 0.01 / 0.02 / 0.05 / 0.1`
  - `visit_threshold` 默认扫：`0 / 1 / 2 / 4`
  - `slow_budget_per_move` 默认扫：`4 / 8 / 16`
  - 先只在 `breakthrough` 上扫，因为它最能体现双阶段潜力。
- 第三步：设一个合理的 slow 触发区间，再比较性能。
  - 目标不是让 `slow_calls` 越多越好，而是先让 `slow_call_ratio` 落到一个可比较区间，默认以 `5%~20%` 为第一阶段目标。
  - 在这个区间内比较：
    - 对 `token_transformer` 的胜率
    - 平均决策时间
    - 对 `pure_mct` 的胜率
- 第四步：如果修完 gate 以后 slow 仍几乎不触发，再回头优化 fast 模型的不确定度。
  - 检查 `variance_head` 输出分布是否塌缩到很小范围。
  - 比较 `variance_head` 与真实误差 `|teacher - fast|` 的相关性。
  - 如果相关性弱，再考虑重训 fast 模型或改 uncertainty loss。

## Test Plan
- 静态验证：
  - `visit` 门控下不应再出现全量 `slow_calls=0`。
  - `combined` 门控下 `visit_ok_rate` 和 `uncertainty_ok_rate` 至少有一个不是近乎 0。
- 功能验证：
  - 在 `breakthrough` 上跑 20 局 smoke，对每组参数输出 `slow_call_ratio` 和 `u_fast` 分位数。
  - 至少找到一组参数使 `slow_call_ratio > 0`。
- 效果验证：
  - 与当前结果对比，确认“slow 真参与后”双阶段方法是否还能保持明显速度优势。
  - 优先接受这样的配置：比 `token_transformer` 更快，且胜率不明显下降；或比当前 two-stage 明显更强，且耗时增幅可控。

## Assumptions
- 默认先不重训模型，先修门控信号与诊断，因为这是当前最可能的主因。
- 默认主战场选 `breakthrough` 做调参，再把最优配置回测到 3 个游戏。
- 默认把“slow 能稳定触发”视为第一阶段成功标准，把“整体强于 transformer”视为第二阶段目标。
- 如果你现在只想选一个最值得先做的动作，推荐顺序是：
  1. 修 `visit_threshold` 的判定对象
  2. 加门控命中率与 `u_fast` 分布日志
  3. 扫 `tau`
  4. 再决定是否重训 fast variance 模型
