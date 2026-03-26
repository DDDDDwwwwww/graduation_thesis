# Two-Stage Experiments

## 1) Reuse previous artifacts

```bash
python experiments4two_stage/build_reused_artifacts.py
```

Default reuse policy:
- Fast net: `token_mlp` from dataset size `1000`
- Slow net: `token_transformer` from dataset size `2000`
- Source priority: `outputs/...` then `Preliminary_experimental/outputs/...`

Output file:
- `outputs4two_stage/artifacts/two_stage_artifacts.json`

## 2) (Optional) Train variance-head fast net with distillation

```bash
python experiments4two_stage/train_fast_variance_net.py \
  --dataset <dataset.jsonl> \
  --teacher-model <slow_model.pt> \
  --teacher-encoder <slow_encoder.json> \
  --output-dir outputs4two_stage/models/fast_variance
```

If you want to use this trained model, update `two_stage_artifacts.json` manually.

## 3) Run experiments

```bash
python experiments4two_stage/run_experiment_ts_a_main_benchmark.py
python experiments4two_stage/run_experiment_ts_b_time_budget.py
python experiments4two_stage/run_experiment_ts_c_search_budget.py
python experiments4two_stage/run_experiment_ts_d_gate_ablation.py
python experiments4two_stage/run_experiment_ts_e_uncertainty_ablation.py
python experiments4two_stage/run_experiment_ts_f_overhead_analysis.py
```

Or batch run:

```bash
python experiments4two_stage/run_all_two_stage_experiments.py --device cpu
```
