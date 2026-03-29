# SDRPV Residual V1 实验报告

## 1. 本轮结论（先给结论）

- 这轮实验是**成功且有正向信号**的：`residual_v1` 在离线指标上已经**超过 baseline evaluator**，并且通过了你设定的 gate。
- 但当前 MCTS 小规模对局结果（`n=4`）在 fixed-sims 和 fixed-time 下都只是 **50% vs 50% 持平**，样本太小，**不能证明线上胜率提升**。
- 因此建议：**继续深入你的计划**，但应先做“扩大小规模验证 + integration 参数调优”，再进入更大规模主实验。

---

## 2. 关键结果解读

### 2.1 离线 residual 指标（核心）

来源：`outputs/experiments/SDRPV_residual_v1_residual_v1_gpu0_20260329_114156/offline_metrics.json`

- Test:
  - `MAE(v_hat, q_t) = 0.17793`
  - `MAE(b, q_t) = 0.18135`
  - `mae_gain_over_b = +0.00342`（约 `+1.89%` 相对改善）
  - `corr(v_hat, q_t) = 0.82856`（高于 `corr(b, q_t)=0.81916`）
  - `rank_corr(v_hat, q_t) = 0.80053`（略高于 `rank_corr(b, q_t)=0.79933`）

判断：
- 你的第 2 条指标要求已经满足：`MAE(v_hat,q_t) < MAE(b,q_t)`，且相关性也提升。
- 改善幅度不大，但方向正确，属于“可继续推进”的结果。

### 2.2 训练过程稳定性

来源：`metrics.json` 与总日志

- `early_stop epoch=19`（patience=5）
- `best_val_loss = 0.02493`
- `test_loss_delta = 0.02540`

判断：
- 训练收敛过程正常，无退化、无异常中断。

### 2.3 小规模 MCTS 对局（fixed-sims / fixed-time）

来源：`outputs/experiments/SDRPV_residual_v1_mcts_smoke_residual_v1_gpu0_20260329_114156/summary/stage_summary.json`

- fixed-sims（120 iters, 0.5s, 4局）：
  - `win_rate_a = 0.5`（2胜2负）
- fixed-time（0.5s, cap=120, 4局）：
  - `win_rate_a = 0.5`（2胜2负）

判断：
- 当前线上结果是“持平”，不是负面，但还不足以支持“已提升”结论。

---

## 3. 与当前计划对照（Day6）

你给的 Day6 与建议项完成度：

1. 新建 residual 训练入口并离线验证：**已完成**
2. 指标增加（corr/rank corr/MAE 对比 baseline）：**已完成**
3. residual 离线有效后接入 MCTS 小规模对局（fixed-sims -> fixed-time）：**已完成**
4. 版本隔离（`residual_v1` 独立产物与日志）：**已完成**

---

## 4. 下一步建议（是否继续）

结论：**应该继续**，但建议按“低风险、可判定”的顺序推进，不要直接跳到大规模主实验。

建议顺序：

1. 扩大小规模对局样本量（优先）
- 把每个设置从 `4` 局提升到 `30~50` 局（至少先到 `20` 局）。
- 目标：先判断 fixed-sims / fixed-time 下是否稳定偏离 50%。

2. 保持同一模型，做 integration 超参小网格
- 先调你当前 MCTS 接入关键超参（例如迭代预算、时间预算、是否 selective 条件）。
- 每次只改 1-2 个变量，避免混淆来源。

3. 若仍长期持平，再做下一阶段增强
- 再考虑你计划里的 selective integration 更细化策略/phase-aware 增强。
- 不建议现在就大范围改 backbone。

---

## 5. 总体判断

这次实验不是“终点胜利”，但已经跨过了最关键门槛：  
**residual 目标在离线确实优于 baseline evaluator，并且流程能端到端跑通。**

所以你的计划路线是成立的，下一步应聚焦“把离线优势转化为稳定线上胜率优势”。

