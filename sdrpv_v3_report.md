# SDRPV V3 实验分析报告

## 1. 结论（先给结论）
- 本次 V3 实验是**有效且有正向信号**的，建议继续按计划进入下一阶段（residual 版本）。
- 你现在不该回去改 backbone；应继续执行你原定路线：`teacher-only -> residual -> selective integration`。

## 2. 本次结果要点

### 2.1 数据转换质量（`sdrpv_dataset_v3_parallel.jsonl.stats.json`）
- 输入样本：5934
- 成功转换：4732
- 跳过：1202（主要是状态去重导致，合理）
- 失败：0
- `q_t.std = 0.3956`（> 0）
- `b.std = 0.3434`（> 0）
- `|q_t-b|.mean = 0.1845`（> 0）

解释：
- 这三个关键值都非 0，说明 teacher/baseline 标签**不再退化为常数**；
- `|q_t-b|` 有明显幅度，残差学习是有信息量的。

### 2.2 teacher-only 训练结果（`SDRPV_teacher_only_v3`）
- 样本数：4732
- 最终测试：`test_loss = 0.04835`，`test_sign_acc = 0.7785`
- 训练曲线从 epoch 1 到 20 持续下降，验证集也同步改善（无明显崩坏）。

解释：
- 和你之前出现的“几秒结束 + loss=0 常数退化”相比，这次是正常学习行为；
- 离线 teacher 拟合能力已经可用，达到了继续推进的门槛。

## 3. 与历史基线的对比判断
- 你历史 `D_dataset_size_sensitivity` 中 token-transformer 离线 `test_sign_acc` 大约在 `0.556`（旧终局监督路线）。
- 本次 teacher-only 达到 `0.778`（注意：任务标签不同，不能直接当最终胜率结论）。

结论：
- 不能直接宣称“实战胜率已提升”，但可以明确判断：  
  **监督信号质量与可学习性显著改善**，满足继续深入实验的条件。

## 4. 是否继续深入实验？
- 建议：**继续**。
- 进入下一步：实现并验证 residual 学习（学习 `q_t - b`），然后再做选择性接入 MCTS。

## 5. 下一步执行建议（按优先级）
1. 新建 residual 训练入口（或在现有入口加 `target = q_t - b` 选项），先做离线验证。
2. 指标增加：
   - `corr(v_hat, q_t)` / rank correlation
   - `MAE(v_hat, q_t)` 与 `MAE(b, q_t)` 的对比（看是否真正超越 baseline evaluator）
3. 仅在 residual 离线有效后，再接入 MCTS 做小规模对局（先 fixed-sims，再 fixed-time）。
4. 保持版本隔离：`v3` 数据、`residual_v1` 模型、独立日志，避免混淆。

## 6. 当前结果的边界与风险提示
- 当前数据只来自 `dataset_200` 且仅 `connectFour`，结论主要是“机制可行”，不是“跨游戏已成立”。
- 当前数据切分是随机样本切分，可能存在同局近邻状态泄漏导致指标偏乐观；后续建议加 `match_id` 分组切分做更严验证。
- `swipl/libtinfo` 仍有 warning，但本次未影响结果。

---

总体评价：  
你这一步做得很好，已经从“标签退化失败”推进到“teacher-only 有效学习”的状态，完全值得继续按计划深入。

