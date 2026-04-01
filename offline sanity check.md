# Offline Sanity Check

## Offline Value Evaluation

Evaluation target: `q_t`  
Evaluation split: `test` split with seed `242` (`n = 474`)  
Compared models: baseline `b`, teacher-only, residual

| Model                         | MAE (lower) | MSE (lower) | Corr (higher) |
| ------------------------------| ----------- | ----------- | ------------- |
| Baseline value b              | 0.1836      | 0.0532      | 0.8137        |
| Teacher-only                  | 0.2038      | 0.0942      | 0.6313        |
| Residual Value Model (SDRPV)  | **0.1788**  | **0.0504**  | **0.8304**    |

## Analysis

The offline sanity check supports the residual route. On the same test split, `Residual Value Model (SDRPV)` achieves the best performance on all three metrics: it reduces MAE from `0.1836` to `0.1788`, reduces MSE from `0.0532` to `0.0504`, and improves correlation from `0.8137` to `0.8304` relative to baseline `b`.

The most important sanity-check signal is that the residual model does not need to replace the baseline estimator from scratch. Instead, it learns a correction on top of `b`, which gives a small but consistent gain. This is exactly the intended behavior of the SDRPV design: keep the cheap baseline prior, then let the learned residual refine it.

By contrast, `Teacher-only` is clearly worse than both `b` and `Residual Value Model (SDRPV)` on this split. Its MAE and MSE are substantially higher, and its correlation drops to `0.6313`. This suggests that directly regressing `q_t` is less stable here than learning a residual around the baseline. So the offline evidence favors the claim that the useful signal is not "replace `b`", but "correct `b`".

## Note

`offline_metrics.json` directly provides the residual-vs-baseline offline comparison for the full residual model. Since it does not include `MSE` or the teacher-only row, the final table was completed by re-evaluating the saved `teacher-only` and `full residual` checkpoints on the same `seed=242` test split from `outputs/datasets/sdrpv_dataset_v3_parallel.jsonl`, using the same `q_t` target definition.

`cheap baseline value `就是字段 `b`，它是在 `convert_dataset_to_sdrpv.py` 里生成的。当前这份数据集 `outputs/datasets/sdrpv_dataset_v3_parallel.jsonl` 实际用的 `cheap evaluator` 是 `shallow_mcts`，不是神经网络。而生成这个baseline value b的`cheap evaluator` 可以准确描述为：`PureMCTAgent with shallow MCTS`, using `32 simulations per state`。而 teacher 则是同一套 search_value(...)，但预算更高，用 600 sims。也就是说：
b = 32-sim shallow MCTS value
q_t = 600-sim teacher search value
所以 residual 学的其实就是：
q_t - b
也就是“高预算搜索值”减去“低预算浅层 MCTS 值”的修正量。