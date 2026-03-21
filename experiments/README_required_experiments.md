# Required Experiments (Section 15)

## Python environment

Use your environment explicitly when running scripts:

```bash
D:/Anaconda/envs/MLcourse/python.exe experiments/run_experiment_a_baseline_strength.py --help
```

## Fixed output layout

Each experiment script now uses the same directory structure under `--out-dir`:

```text
<out-dir>/
  meta/
    run_manifest.json
  raw/
    matches.json
  summary/
    by_game.json
    by_game.csv
    cross_game.json
    cross_game.csv
  metrics/               # optional, used by some experiments (e.g., D/E)
  artifacts/             # optional, used by some experiments (e.g., D/G/I)
  datasets/              # optional, used by data-generating experiments
  models/                # optional, used by training experiments
```

## Artifact map format

For scripts that accept `--artifacts`, prepare a JSON file like:

```json
{
  "fact_mlp": {
    "model_path": "outputs/models/fact_mlp/model.pt",
    "encoder_config_path": "outputs/models/fact_mlp/encoder.json",
    "vocab_path": "outputs/models/fact_mlp/vocab.json"
  },
  "token_mlp": {
    "model_path": "outputs/models/token_mlp/model.pt",
    "encoder_config_path": "outputs/models/token_mlp/encoder.json",
    "vocab_path": null
  },
  "token_transformer": {
    "model_path": "outputs/models/token_transformer/model.pt",
    "encoder_config_path": "outputs/models/token_transformer/encoder.json",
    "vocab_path": null
  }
}
```

## Scripts

- `run_experiment_a_baseline_strength.py`
- `run_experiment_b_time_budget.py`
- `run_experiment_c_search_budget.py`
- `run_experiment_d_dataset_size.py`
- `run_experiment_e_encoder_model_ablation.py`
- `run_experiment_f_cache_performance.py`
- `run_experiment_g_single_vs_multi.py`
- `run_experiment_h_multi_game_benchmark.py`
- `run_experiment_i_cross_game_generalization.py`
- `run_all_required_experiments.py`

## Useful helpers

- `generate_dataset.py` (single-game self-play dataset)
- `generate_multigame_dataset.py` (multi-game mixed dataset)
- `train_value_model.py` (supports `fact_vector+mlp`, `board_token+mlp`, `board_token+transformer`)

## Example

```bash
D:/Anaconda/envs/MLcourse/python.exe experiments/run_experiment_a_baseline_strength.py --artifacts outputs/artifacts.json
D:/Anaconda/envs/MLcourse/python.exe experiments/run_experiment_d_dataset_size.py --game games/connectFour.kif
D:/Anaconda/envs/MLcourse/python.exe experiments/run_all_required_experiments.py --artifacts outputs/artifacts.json --experiments A B C H
```
