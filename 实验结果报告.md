# 实验结果报告

## 1. 报告说明
- 报告生成时间：2026-03-27。
- 数据来源：`experiments4two_stage` 与 `outputs4two_stage` 下的实验脚本、`summary/*.csv`、`summary/*.json`、`meta/run_manifest.json`、`artifacts/two_stage_artifacts.json`、`models/fast_variance_main/metrics.json`。
- 主要实验：
  - `TS_A_main_benchmark`：双阶段方法主基准对比。
  - `TS_B_time_budget_sensitivity`：时间预算敏感性。
  - `TS_C_search_budget_sensitivity`：搜索预算敏感性。
  - `TS_D_gate_ablation`：门控策略消融。
  - `TS_E_uncertainty_ablation`：不确定性估计方式消融。
  - `TS_F_overhead_analysis`：推理开销分析。
- 复用模型来源：
  - fast 基线：`token_mlp@size_1000`
  - slow 基线：`token_transformer@size_2000`
  - 双阶段 agent 的 fast 模型使用 `outputs4two_stage/models/fast_variance_main`，slow 模型复用 `token_transformer@size_2000`。

## 2. 实验设置概览
- 主基准与开销分析覆盖 3 个游戏：`ticTacToe`、`connectFour`、`breakthrough`。
- 敏感性实验与消融实验主要在 `breakthrough` 上进行。
- 主配置（来自 `TS_A_main_benchmark/meta/run_manifest.json`）：
  - `rounds=20`
  - `playclock=0.7s`
  - `iterations=120`
  - `tau=0.15`
  - `visit_threshold=4`
  - `slow_budget_per_move=16`
  - `uncertainty_type=variance_head`
  - `gate_type=combined`
- fast variance 模型训练指标（`outputs4two_stage/models/fast_variance_main/metrics.json`）：
  - `best_val_nll=-4.4014`
  - `test_nll=-4.4027`
  - `test_mse=0.9837`

## 3. 总体结论
- 当前结果说明：**双阶段方法在总体上优于较弱的 `token_mlp` 基线，但尚未稳定超过 `token_transformer` 单模型 MCTS。**
- 从跨游戏均值看：
  - `two_stage_neural_mcts vs pure_mct` 的平均胜率为 **0.417**，高于 `neural_mcts:token_mlp vs pure_mct` 的 **0.350**，但低于 `neural_mcts:token_transformer vs pure_mct` 的 **0.450**。
  - 双阶段方法直接对战单模型时，`vs token_mlp` 的胜率为 **0.350**，`vs token_transformer` 的胜率为 **0.367**，暂未表现出显著优势。
- **breakthrough 是双阶段方法最有潜力的游戏。** 在该游戏上，双阶段方法对 `pure_mct` 胜率达到 **0.700**，高于 `token_transformer` 的 **0.650**。
- **效率方面，双阶段方法明显快于 transformer-MCTS，接近 MLP-MCTS。** 在对 random 的开销分析中：
  - 双阶段方法平均决策时间 **0.192s/步**；
  - `token_transformer` 为 **0.310s/步**；
  - `token_mlp` 为 **0.189s/步**。
- 但需要特别指出：**所有实验中 `slow_calls` 都为 0，说明当前参数与模型下 slow 分支没有被触发。** 因此，这批“双阶段”结果在实际行为上更接近“带不确定性头的 fast-only MCTS”，而不是真正发生 fast/slow 切换的版本。

## 4. 主基准实验 A（TS_A_main_benchmark）
实验设置：3 个游戏、每组 20 局、`playclock=0.7s`、`iterations=120`。

### 4.1 跨游戏均值结果
- `random vs pure_mct`：胜率 **0.117**，说明纯 MCTS 依然是强基线。
- `neural_mcts:token_mlp vs pure_mct`：胜率 **0.350**。
- `neural_mcts:token_transformer vs pure_mct`：胜率 **0.450**。
- `two_stage_neural_mcts vs pure_mct`：胜率 **0.417**。
- `two_stage_neural_mcts vs neural_mcts:token_mlp`：胜率 **0.350**。
- `two_stage_neural_mcts vs neural_mcts:token_transformer`：胜率 **0.367**。

结论：
- 双阶段方法相较 `token_mlp` 有一定提升，但总体仍弱于 `token_transformer`。
- 如果把这组结果作为主结论，更稳妥的说法是：**双阶段方法在“效率接近 MLP”的前提下，取得了介于 MLP-MCTS 和 transformer-MCTS 之间的对局强度。**

### 4.2 分游戏结果
#### `ticTacToe`
- `two_stage vs pure_mct`：胜率 **0.100**。
- `two_stage vs token_transformer`：胜率 **0.350**。
- 说明双阶段方法在简单小游戏上没有体现优势。

#### `connectFour`
- `two_stage vs pure_mct`：胜率 **0.450**。
- `two_stage vs token_transformer`：胜率 **0.500**。
- 说明双阶段方法在 `connectFour` 上与 transformer 基本打平，但还没有明显超出。

#### `breakthrough`
- `two_stage vs pure_mct`：胜率 **0.700**。
- `token_transformer vs pure_mct`：胜率 **0.650**。
- `two_stage vs token_transformer`：胜率 **0.250**。
- 这表明双阶段方法在复杂博弈上的潜力更大，但对强神经搜索对手的优势仍不稳定。

## 5. 时间预算敏感性 B（TS_B_time_budget_sensitivity）
实验设置：`breakthrough`，时间预算为 `0.2s / 0.7s / 1.5s`。

结果：
- `token_transformer vs token_mlp` 胜率：**0.70 -> 0.65 -> 0.60**。
- `two_stage vs token_transformer` 胜率：**0.35 -> 0.40 -> 0.40**。
- 双阶段方法平均决策时间：**0.103s -> 0.350s -> 0.501s**。
- 双阶段方法平均 fast 调用次数：**2471.6 -> 4013.7 -> 11421.8**。

解释：
- 增加时间预算确实让双阶段方法做了更多 fast 评估，但胜率只从 0.35 小幅提升到 0.40，边际收益有限。
- 由于 `slow_call_ratio` 始终为 0，时间预算增加本质上只是给 fast 模型更多搜索次数，而没有带来 slow 模型的补强效果。

## 6. 搜索预算敏感性 C（TS_C_search_budget_sensitivity）
实验设置：`breakthrough`，搜索迭代为 `50 / 120 / 300`，`playclock=0.7s`。

结果：
- `two_stage vs token_transformer` 胜率：**0.75 -> 0.50 -> 0.45**。
- 双阶段方法平均决策时间：**0.219s -> 0.350s -> 0.352s**。
- 双阶段方法平均 fast 调用次数：**2231.0 -> 5246.2 -> 6769.1**。

解释：
- 在较小搜索预算 `50` 下，双阶段方法反而取得最好结果（0.75）。
- 随着搜索迭代继续增大，胜率下降到 0.50 和 0.45，说明当前 fast-only 行为模式下，更多搜索并没有稳定转化为更强对局表现。
- 这也暗示当前快模型的评估偏差可能会在深搜索中被放大，而 slow 分支又没有参与纠偏。

## 7. 门控消融 D（TS_D_gate_ablation）
实验设置：`breakthrough`，比较 `uncertainty`、`visit`、`uncertainty_visit`、`combined` 四种门控。

结果：
- `uncertainty`：胜率 **0.60**。
- `visit`：胜率 **0.15**。
- `uncertainty_visit`：胜率 **0.25**。
- `combined`：胜率 **0.60**。
- 四种设置下 `mean_slow_call_ratio_a` 均为 **0.0**。

解释：
- 从表面结果看，`uncertainty` 和 `combined` 最好，`visit` 最差。
- 但由于所有设置都没有触发 slow 分支，**这些差异不能直接解释为门控策略本身优劣**；更可能混入了 20 局样本量下的随机波动。
- 因此，这组实验目前更适合写成“现象观察”，不宜下过强结论。

## 8. 不确定性估计消融 E（TS_E_uncertainty_ablation）
实验设置：`breakthrough`，比较 `margin` 与 `variance_head`。

结果：
- `margin`：胜率 **0.30**。
- `variance_head`：胜率 **0.45**。
- 二者平均决策时间接近，分别为 **0.347s** 和 **0.349s**。
- 二者 `mean_slow_call_ratio_a` 均为 **0.0**。

解释：
- 在当前结果里，`variance_head` 明显优于 `margin`，这说明即使 slow 分支未触发，方差头本身可能仍通过 fast 模型训练质量改善了价值评估。
- 不过，这个优势依然是在 fast-only 行为下观察到的，尚不能证明它更适合真实的 fast/slow 切换框架。

## 9. 开销分析 F（TS_F_overhead_analysis）
实验设置：3 个游戏，分析各方法对 random 的速度与基础强度。

结果：
- `two_stage vs random`：
  - 胜率 **0.883**
  - 平均决策时间 **0.192s/步**
- `token_transformer vs random`：
  - 胜率 **0.817**
  - 平均决策时间 **0.310s/步**
- `token_mlp vs random`：
  - 胜率 **0.850**
  - 平均决策时间 **0.189s/步**

解释：
- 双阶段方法相对 transformer-MCTS 将平均决策时间降低了约 **38%**（`(0.310-0.192)/0.310`），同时对 random 的胜率还略高。
- 双阶段方法与 MLP-MCTS 速度几乎相同，但强度略高（0.883 vs 0.850）。
- 因此，这批结果最有价值的结论其实在“**速度-性能折中**”上，而不是“绝对最强胜率”。

## 10. 结果局限与原因分析
- 最重要的限制是：**所有 raw 结果中都没有出现非零 `slow_calls`。** 这意味着：
  - 当前 `tau=0.15`、`visit_threshold=4`、`slow_budget_per_move=16` 的组合没有真正触发 slow 模型；
  - B/C/D/E 四组实验本质上主要在比较 fast 分支条件下的搜索表现，而不是完整双阶段调度策略。
- 因此，当前实验更准确的结论应是：
  - 双阶段框架的 fast 端设计是有效的；
  - 但 slow 端门控机制尚未在实验中体现出实际作用。
- 样本量方面，大多数单组设置只有 20 局，对 `breakthrough` 这类波动较大的博弈来说，统计稳定性仍然有限。

## 11. 可直接写入论文的简短结论
本组 two-stage 实验表明：在当前实现与参数设置下，双阶段神经 MCTS 在整体对局强度上处于 `token_mlp` 与 `token_transformer` 两种单模型神经 MCTS 之间，其中在 `breakthrough` 上表现出较好的潜力；同时，其平均决策开销显著低于 transformer-MCTS，并与 MLP-MCTS 基本持平，体现出较好的速度-性能折中。另一方面，所有实验中 slow 分支均未被触发，说明当前结果尚不能完全验证 fast/slow 协同调度机制本身，后续需要进一步调整门控阈值或不确定性标定，使 slow 模型真正参与搜索决策，才能完成对完整双阶段框架的有效评估。
