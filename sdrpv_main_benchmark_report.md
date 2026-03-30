# SDRPV Main Benchmark 报告（Residual + Selective，非 Phase-Aware）

## 1. 结论（先给结论）

- 这轮主实验是有价值的，且结果足以支持继续推进当前方向。
- 当前最合理的主方法不是 `sel_cap20`，而是 `value_full`：
  - `value_full` 在本轮三游戏 benchmark 下已经整体超过 `pure_mct`；
  - `sel_cap20` 相比 `value_full` 明显退化，暂不适合作为下一步主配置。
- 当前方法的主要短板集中在 `ticTacToe`：
  - `connectFour` 与 `breakthrough` 已经表现出明显优势；
  - `ticTacToe` 胜率远低于 `0.5`，是当前跨游戏稳定性的主要瓶颈。
- 因此，下一步不应继续主推 selective 调参，而应：
  - 先以 `value_full` 作为当前主方法进入正式 baseline 对比；
  - 并行启动 `ticTacToe` 专项补强；
  - 暂不进入 phase-aware。

---

## 2. 实验设置

本轮运行配置来自 `run_manifest.json`：

- 模型：`outputs/experiments/SDRPV_residual_v1_residual_v1_gpu0_20260329_114156`
- 游戏：`ticTacToe`、`connectFour`、`breakthrough`
- rounds：`50`
- seeds：`42, 142, 242`
- 评测方式：`fixed-sims` + `fixed-time`
- 对比配置：
  - `value_full`：不做 selective，直接使用 value evaluator
  - `sel_cap20`：`selective_max_neural_evals_per_move=20`，`alpha=1.0`

---

## 3. 总体结果（跨游戏汇总）

来源：`summary/aggregate_summary.json`

### 3.1 `value_full`
- fixed-sims：`0.5067`（`450` 局）
- fixed-time：`0.5333`（`450` 局）

### 3.2 `sel_cap20`
- fixed-sims：`0.3867`（`450` 局）
- fixed-time：`0.4200`（`450` 局）

### 3.3 对比差值（`sel_cap20 - value_full`）
- fixed-sims：`-0.1200`
- fixed-time：`-0.1133`

结论：在当前多游戏主实验设置下，`sel_cap20` 总体退化，而 `value_full` 已经在当前 benchmark 定义下整体超过 `pure_mct`。

---

## 4. 分游戏结果（关键）

按 seed 平均后的分游戏胜率如下：

### 4.1 fixed-sims
- `ticTacToe`：`value_full=0.1400`，`sel_cap20=0.0133`
- `connectFour`：`value_full=0.6933`，`sel_cap20=0.6467`
- `breakthrough`：`value_full=0.6867`，`sel_cap20=0.5000`

### 4.2 fixed-time
- `ticTacToe`：`value_full=0.1267`，`sel_cap20=0.0133`
- `connectFour`：`value_full=0.7333`，`sel_cap20=0.7467`
- `breakthrough`：`value_full=0.7400`，`sel_cap20=0.5000`

结论：
- `value_full` 已在 `connectFour` 和 `breakthrough` 上明显优于 `pure_mct`；
- `sel_cap20` 仅在 `connectFour` 的 fixed-time 下有小幅优势；
- `ticTacToe` 是当前唯一明显没有过 `0.5` 的游戏，也是下一步最值得单独修复的对象。

---

## 5. 与方案路线对照（`search_distilled_residual_value_plan.md`）

当前进度与原计划基本一致：

1. `teacher-only -> residual` 已完成并验证有效；
2. 已进入 `selective integration` 阶段并完成对照评测；
3. 当前尚未引入 `phase-aware`，符合“后置增强”的原始规划。

本轮结果支持以下判断：
- `residual` 路线本身是可行的；
- integration 方式对 GGP 下的最终效果影响很大；
- `selective integration` 的收益具有明显游戏依赖性；
- 当前最稳妥的主线应先落在 `value_full`，而不是继续推动统一 selective 配置。

---

## 6. 风险解释（为什么会出现这种结果）

结合日志与指标，当前 `sel_cap20` 可能存在以下问题：

- 额外神经评估开销带来搜索预算损失；
- 同一组 selective 参数无法同时适配三款游戏；
- `connectFour` 上的局部收益不足以抵消 `ticTacToe` 和 `breakthrough` 上的退化；
- 在当前预算下，`value_full` 的稳定性明显高于 `sel_cap20`。

这说明当前阶段的关键问题不是“模型是否已经有信号”，而是“哪种 integration 方式最适合把信号稳定转化为胜率”。

---

## 7. 下一步执行建议（可直接做）

### 7.1 当前主方法定位（推荐）

1. 将 `value_full` 明确作为当前主方法继续推进。
2. 将 `sel_cap20` 退回为 integration ablation，不再作为下一步主配置。
3. 论文叙述上明确区分两件事：
   - 当前 `residual_v1 + value_full` 已经在本轮 benchmark 下整体超过 `pure_mct`；
   - 但其跨游戏稳定性仍不完整，主要短板集中在 `ticTacToe`。

### 7.2 下一步实验主线（短周期）

1. 不改 backbone，不引入 phase-aware，不再继续主推 selective 调参。
2. 直接使用当前最佳版本 `value_full` 开始正式 baseline 对比，比较对象为：
   - `pure_mct`
   - `outputs/experiments/D_dataset_size_sensitivity/models/size_200/token_transformer/model.pt`
   - `outputs/experiments/D_dataset_size_sensitivity/models/size_200/fact_mlp/model.pt`
   - `HeuristicMCTS`
3. 对比设置保持与本轮主 benchmark 一致，继续报告：
   - fixed-sims
   - fixed-time
   - 三游戏平均结果
   - 分游戏结果
   - 多 seed 平均结果
4. 在 baseline 对比并行，单独开一条 `ticTacToe` 专项提升线，目标不是重做整套方法，而是把唯一明显低于 `0.5` 的游戏补强。

### 7.3 `ticTacToe` 专项提升目标

1. 将目标收敛为：先把 `ticTacToe` 胜率提升到接近或超过 `0.5`。
2. 提升优先级建议为：
   - 先检查该游戏下的 value 引导是否系统性误导搜索；
   - 再检查预算设置、编码表示、训练样本覆盖是否对小盘面游戏不友好；
   - 如有必要，可只对 `ticTacToe` 做轻量 game-specific 调整，而不是改全局方案。
3. 只有当 `ticTacToe` 补强版本确实带来稳定收益时，再回到统一三游戏 benchmark 做补跑。

### 7.4 停止条件（防止无限调参）

- 若当前 `value_full` 在四个 baseline 对比中已经形成清晰相对定位，则先固化主结果与论文叙述，不以“继续调参”作为唯一目标。
- 若 `ticTacToe` 在 1~2 轮轻量修正后仍无明显改善，则停止对该游戏做大规模结构改动，保留其为当前方法的已知短板，并转入结果整理与分析写作。
- 后续只有在新增改动同时改善 `ticTacToe` 且不破坏 `connectFour` / `breakthrough` 表现时，才值得进入下一轮统一大 benchmark。

---

## 8. 最终判断

这轮主实验是值得保留的有效结果，但它的意义不是“全局 selective 已经胜出”，而是：

> 在当前 GGP 框架下，`residual` 路线已经能够整体超过 `pure_mct`；但 `selective integration` 的收益具有显著游戏依赖性，当前更稳妥的主方法应是 `value_full`，而后续提升重点应集中在 `ticTacToe` 的专项补强与正式 baseline 对比。

这个结论完全可以成为论文中的核心研究结论，而不是失败信号。