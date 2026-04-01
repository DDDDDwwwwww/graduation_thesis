from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


NEURAL_KEYS = ("token_transformer")


def _runtime_imports():
    from agents import (  # noqa: WPS433
        HeuristicMCTSAgent,
        NeuralValueMCTSAgent,
        PureMCTAgent,
    )
    from ggp_statemachine import GameStateMachine  # noqa: WPS433

    return {
        "HeuristicMCTSAgent": HeuristicMCTSAgent,
        "NeuralValueMCTSAgent": NeuralValueMCTSAgent,
        "PureMCTAgent": PureMCTAgent,
        "GameStateMachine": GameStateMachine,
    }


@dataclass(frozen=True)
class ValueArtifact:
    model_path: str
    encoder_config_path: str
    vocab_path: str | None = None


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def init_output_layout(
    exp_name: str,
    out_dir: str | Path,
    args: argparse.Namespace | dict | None = None,
) -> dict[str, Path]:
    root = ensure_dir(out_dir)
    layout = {
        "root": root,
        "meta": ensure_dir(root / "meta"),
        "raw": ensure_dir(root / "raw"),
        "summary": ensure_dir(root / "summary"),
        "metrics": ensure_dir(root / "metrics"),
        "artifacts": ensure_dir(root / "artifacts"),
        "datasets": ensure_dir(root / "datasets"),
        "models": ensure_dir(root / "models"),
    }
    payload = {
        "experiment": exp_name,
        "root": str(root),
    }
    if args is not None:
        if isinstance(args, argparse.Namespace):
            payload["args"] = vars(args)
        elif isinstance(args, dict):
            payload["args"] = dict(args)
        else:
            payload["args"] = str(args)
    write_json(layout["meta"] / "run_manifest.json", payload)
    return layout


def default_games(multi: bool = True) -> list[str]:
    if multi:
        return [
            "games/hex.kif",
            "games/connectFour.kif",
            "games/breakthrough.kif",
        ]
    return ["games/connectFour.kif"]


def load_artifact_map(path: str | Path) -> dict[str, ValueArtifact]:
    payload = read_json(path)
    out: dict[str, ValueArtifact] = {}
    for key in NEURAL_KEYS:
        if key not in payload:
            raise ValueError(f"artifact map missing required key: {key}")
        row = payload[key]
        out[key] = ValueArtifact(
            model_path=str(row["model_path"]),
            encoder_config_path=str(row["encoder_config_path"]),
            vocab_path=(None if row.get("vocab_path") in (None, "") else str(row.get("vocab_path"))),
        )
    return out


def run_cmd(args: list[str], cwd: str | Path | None = None) -> None:
    subprocess.run(args, cwd=str(cwd or ROOT), check=True)


def generate_dataset(
    game: str,
    n_games: int,
    output_path: str | Path,
    agent: str = "mixed_heuristic_pure",
    heuristic_ratio: float = 0.8,
    iterations: int = 100,
    playclock: float = 1.0,
    seed: int = 42,
    sampling_mode: str = "all_states",
    sample_rate: float = 0.4,
) -> None:
    run_cmd(
        [
            sys.executable,
            str(ROOT / "experiments" / "generate_dataset.py"),
            "--game",
            str(game),
            "--agent",
            str(agent),
            "--heuristic-ratio",
            str(heuristic_ratio),
            "--n-games",
            str(int(n_games)),
            "--iterations",
            str(int(iterations)),
            "--playclock",
            str(float(playclock)),
            "--seed",
            str(int(seed)),
            "--sampling-mode",
            str(sampling_mode),
            "--sample-rate",
            str(float(sample_rate)),
            "--output",
            str(output_path),
        ]
    )


def train_value_model_artifact(
    dataset_path: str | Path,
    output_dir: str | Path,
    encoder: str,
    model: str,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
) -> None:
    run_cmd(
        [
            sys.executable,
            str(ROOT / "experiments" / "train_value_model.py"),
            "--dataset",
            str(dataset_path),
            "--encoder",
            str(encoder),
            "--model",
            str(model),
            "--epochs",
            str(int(epochs)),
            "--batch-size",
            str(int(batch_size)),
            "--learning-rate",
            str(float(learning_rate)),
            "--weight-decay",
            str(float(weight_decay)),
            "--seed",
            str(int(seed)),
            "--output-dir",
            str(output_dir),
            "--device",
            str(device),
        ]
    )


def _build_agent(
    agent_key: str,
    role: str,
    seed: int,
    artifacts: dict[str, ValueArtifact] | None,
    iterations: int,
    device: str,
    agent_overrides: dict[str, dict] | None = None,
):
    overrides = dict((agent_overrides or {}).get(agent_key, {}))
    runtime = _runtime_imports()
    if agent_key == "random":
        return runtime["RandomAgent"](name=f"random_{role}", role=role, seed=seed)
    if agent_key == "pure_mct":
        return runtime["PureMCTAgent"](name=f"pure_{role}", role=role, iterations=iterations, seed=seed)
    if agent_key == "heuristic_mcts":
        return runtime["HeuristicMCTSAgent"](name=f"heur_{role}", role=role, iterations=iterations, seed=seed)
    if agent_key.startswith("value_greedy:"):
        if artifacts is None:
            raise ValueError("ValueGreedy agent requires artifacts map.")
        cfg = agent_key.split(":", 1)[1]
        art = artifacts[cfg]
        return runtime["ValueGreedyAgent"].from_artifacts(
            name=f"vg_{cfg}_{role}",
            role=role,
            model_path=art.model_path,
            vocab_path=art.vocab_path,
            encoder_config_path=art.encoder_config_path,
            device=device,
            seed=seed,
            debug=False,
        )
    if agent_key.startswith("neural_mcts:"):
        if artifacts is None:
            raise ValueError("NeuralValueMCTS agent requires artifacts map.")
        cfg = agent_key.split(":", 1)[1]
        art = artifacts[cfg]
        fallback_legal_threshold = overrides.pop("fallback_legal_threshold", None)
        evaluator_mode = str(overrides.pop("evaluator_mode", "value"))
        selective_max_neural_evals_per_move = overrides.pop("selective_max_neural_evals_per_move", None)
        selective_legal_move_threshold = overrides.pop("selective_legal_move_threshold", None)
        selective_alpha = overrides.pop("selective_alpha", 1.0)
        selective_rollout_depth_limit = overrides.pop("selective_rollout_depth_limit", 64)
        if overrides:
            raise ValueError(f"Unknown neural_mcts overrides for {agent_key}: {sorted(overrides.keys())}")
        return runtime["NeuralValueMCTSAgent"].from_artifacts(
            name=f"nm_{cfg}_{role}",
            role=role,
            model_path=art.model_path,
            vocab_path=art.vocab_path,
            encoder_config_path=art.encoder_config_path,
            iterations=iterations,
            exploration_c=1.4,
            discount_factor=0.99,
            device=device,
            evaluator_mode=evaluator_mode,
            seed=seed,
            fallback_legal_threshold=fallback_legal_threshold,
            selective_max_neural_evals_per_move=selective_max_neural_evals_per_move,
            selective_legal_move_threshold=selective_legal_move_threshold,
            selective_alpha=selective_alpha,
            selective_rollout_depth_limit=selective_rollout_depth_limit,
        )
    raise ValueError(f"Unknown agent key: {agent_key}")


def run_single_match(
    game_file: str,
    agent_a_key: str,
    agent_b_key: str,
    a_is_first_role: bool,
    playclock: float,
    iterations: int,
    seed: int,
    cache_enabled: bool,
    artifacts: dict[str, ValueArtifact] | None,
    device: str,
    agent_overrides: dict[str, dict] | None = None,
) -> dict:
    runtime = _runtime_imports()
    game = runtime["GameStateMachine"](game_file, cache_enabled=cache_enabled)
    all_roles = [str(r) for r in game.get_roles()]
    init_state = game.get_initial_state()
    roles = []
    for r in all_roles:
        if r in roles:
            continue
        if game.get_legal_moves(init_state, r):
            roles.append(r)
    if len(roles) != 2:
        raise ValueError(
            "Only 2-player games are supported for head-to-head evaluation. "
            f"game={game_file}, all_roles={all_roles}, active_roles={roles}"
        )

    role_a = roles[0] if a_is_first_role else roles[1]
    role_b = roles[1] if a_is_first_role else roles[0]

    agent_a = _build_agent(
        agent_a_key,
        role_a,
        seed=seed + 11,
        artifacts=artifacts,
        iterations=iterations,
        device=device,
        agent_overrides=agent_overrides,
    )
    agent_b = _build_agent(
        agent_b_key,
        role_b,
        seed=seed + 29,
        artifacts=artifacts,
        iterations=iterations,
        device=device,
        agent_overrides=agent_overrides,
    )
    agents = {role_a: agent_a, role_b: agent_b}

    state = init_state
    ply = 0
    decision_seconds = {role_a: [], role_b: []}
    eval_calls = {
        role_a: {"eval_calls_total": 0, "eval_calls_neural": 0, "eval_calls_fallback": 0, "eval_calls_mixed": 0},
        role_b: {"eval_calls_total": 0, "eval_calls_neural": 0, "eval_calls_fallback": 0, "eval_calls_mixed": 0},
    }

    game.clear_caches()
    game.reset_perf_stats()

    while not game.is_terminal(state):
        joint = {}
        for role in roles:
            legal = game.get_legal_moves(state, role)
            t0 = time.perf_counter()
            move = agents[role].select_action(game, state, legal, time_limit=playclock)
            decision_seconds[role].append(time.perf_counter() - t0)
            stats_fn = getattr(agents[role], "get_last_search_stats", None)
            if callable(stats_fn):
                stats = stats_fn() or {}
                for key in ("eval_calls_total", "eval_calls_neural", "eval_calls_fallback", "eval_calls_mixed"):
                    eval_calls[role][key] += int(stats.get(key, 0))
            joint[role] = move
        state = game.get_next_state(state, joint)
        ply += 1

    scores = {r: float(game.get_goal(state, r)) for r in roles}
    perf = game.get_perf_stats()
    max_score = max(scores.values())
    winners = [r for r, s in scores.items() if s == max_score]

    return {
        "game": Path(game_file).stem,
        "game_file": game_file,
        "role_order": roles,
        "agent_a_key": agent_a_key,
        "agent_b_key": agent_b_key,
        "role_a": role_a,
        "role_b": role_b,
        "score_a": scores[role_a],
        "score_b": scores[role_b],
        "winner": "draw" if len(winners) != 1 else ("a" if winners[0] == role_a else "b"),
        "ply_count": ply,
        "avg_decision_sec_a": statistics.fmean(decision_seconds[role_a]) if decision_seconds[role_a] else 0.0,
        "avg_decision_sec_b": statistics.fmean(decision_seconds[role_b]) if decision_seconds[role_b] else 0.0,
        "avg_eval_calls_total_a": (eval_calls[role_a]["eval_calls_total"] / max(1, ply)),
        "avg_eval_calls_total_b": (eval_calls[role_b]["eval_calls_total"] / max(1, ply)),
        "avg_eval_calls_neural_a": (eval_calls[role_a]["eval_calls_neural"] / max(1, ply)),
        "avg_eval_calls_neural_b": (eval_calls[role_b]["eval_calls_neural"] / max(1, ply)),
        "avg_eval_calls_fallback_a": (eval_calls[role_a]["eval_calls_fallback"] / max(1, ply)),
        "avg_eval_calls_fallback_b": (eval_calls[role_b]["eval_calls_fallback"] / max(1, ply)),
        "avg_eval_calls_mixed_a": (eval_calls[role_a]["eval_calls_mixed"] / max(1, ply)),
        "avg_eval_calls_mixed_b": (eval_calls[role_b]["eval_calls_mixed"] / max(1, ply)),
        "legal_calls": int(perf["legal_calls"]),
        "legal_cache_hits": int(perf["legal_cache_hits"]),
        "next_calls": int(perf["next_calls"]),
        "next_cache_hits": int(perf["next_cache_hits"]),
        "cache_enabled": bool(cache_enabled),
    }


def run_series(
    game_file: str,
    agent_a_key: str,
    agent_b_key: str,
    rounds: int,
    playclock: float,
    iterations: int,
    seed: int,
    cache_enabled: bool,
    artifacts: dict[str, ValueArtifact] | None,
    device: str,
    swap_roles: bool = True,
    agent_overrides: dict[str, dict] | None = None,
) -> list[dict]:
    rows = []
    game_name = Path(game_file).stem
    for i in range(int(rounds)):
        if swap_roles:
            first = (i % 2 == 0)
        else:
            first = True
        match_seed = seed + i * 101
        t0 = time.perf_counter()
        row = run_single_match(
            game_file=game_file,
            agent_a_key=agent_a_key,
            agent_b_key=agent_b_key,
            a_is_first_role=first,
            playclock=playclock,
            iterations=iterations,
            seed=match_seed,
            cache_enabled=cache_enabled,
            artifacts=artifacts,
            device=device,
            agent_overrides=agent_overrides,
        )
        rows.append(row)
        elapsed = time.perf_counter() - t0
        print(
            "[progress][match] "
            f"game={game_name} pair={agent_a_key}__vs__{agent_b_key} "
            f"round={i + 1}/{int(rounds)} winner={row['winner']} "
            f"score_a={row['score_a']:.1f} score_b={row['score_b']:.1f} "
            f"ply={row['ply_count']} elapsed_sec={elapsed:.2f}",
            flush=True,
        )
    return rows


def summarize_series(rows: list[dict]) -> dict:
    if not rows:
        return {}

    wins = sum(1 for r in rows if r["winner"] == "a")
    losses = sum(1 for r in rows if r["winner"] == "b")
    draws = sum(1 for r in rows if r["winner"] == "draw")
    n = len(rows)

    legal_calls = sum(int(r["legal_calls"]) for r in rows)
    legal_hits = sum(int(r["legal_cache_hits"]) for r in rows)
    next_calls = sum(int(r["next_calls"]) for r in rows)
    next_hits = sum(int(r["next_cache_hits"]) for r in rows)

    return {
        "game": rows[0]["game"],
        "agent_a_key": rows[0]["agent_a_key"],
        "agent_b_key": rows[0]["agent_b_key"],
        "n_matches": n,
        "wins_a": wins,
        "losses_a": losses,
        "draws": draws,
        "win_rate_a": wins / n,
        "draw_rate": draws / n,
        "avg_score_a": statistics.fmean(float(r["score_a"]) for r in rows),
        "avg_score_b": statistics.fmean(float(r["score_b"]) for r in rows),
        "avg_game_length": statistics.fmean(int(r["ply_count"]) for r in rows),
        "avg_decision_sec_a": statistics.fmean(float(r["avg_decision_sec_a"]) for r in rows),
        "avg_decision_sec_b": statistics.fmean(float(r["avg_decision_sec_b"]) for r in rows),
        "avg_eval_calls_total_a": statistics.fmean(float(r.get("avg_eval_calls_total_a", 0.0)) for r in rows),
        "avg_eval_calls_total_b": statistics.fmean(float(r.get("avg_eval_calls_total_b", 0.0)) for r in rows),
        "avg_eval_calls_neural_a": statistics.fmean(float(r.get("avg_eval_calls_neural_a", 0.0)) for r in rows),
        "avg_eval_calls_neural_b": statistics.fmean(float(r.get("avg_eval_calls_neural_b", 0.0)) for r in rows),
        "avg_eval_calls_fallback_a": statistics.fmean(float(r.get("avg_eval_calls_fallback_a", 0.0)) for r in rows),
        "avg_eval_calls_fallback_b": statistics.fmean(float(r.get("avg_eval_calls_fallback_b", 0.0)) for r in rows),
        "avg_eval_calls_mixed_a": statistics.fmean(float(r.get("avg_eval_calls_mixed_a", 0.0)) for r in rows),
        "avg_eval_calls_mixed_b": statistics.fmean(float(r.get("avg_eval_calls_mixed_b", 0.0)) for r in rows),
        "legal_cache_hit_rate": (legal_hits / legal_calls) if legal_calls else 0.0,
        "next_cache_hit_rate": (next_hits / next_calls) if next_calls else 0.0,
        "cache_enabled": bool(rows[0].get("cache_enabled", True)),
    }


def run_match_grid(
    games: list[str],
    pairs: list[tuple[str, str]],
    rounds: int,
    playclock: float,
    iterations: int,
    seed: int,
    cache_enabled: bool,
    artifacts: dict[str, ValueArtifact] | None,
    device: str,
    agent_overrides: dict[str, dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    raw_rows: list[dict] = []
    summary_rows: list[dict] = []

    total_groups = len(games) * len(pairs)
    group_idx = 0
    for game in games:
        game_name = Path(game).stem
        for a, b in pairs:
            group_idx += 1
            print(
                "[progress][group-start] "
                f"{group_idx}/{total_groups} game={game_name} pair={a}__vs__{b}",
                flush=True,
            )
            t0 = time.perf_counter()
            series = run_series(
                game_file=game,
                agent_a_key=a,
                agent_b_key=b,
                rounds=rounds,
                playclock=playclock,
                iterations=iterations,
                seed=seed,
                cache_enabled=cache_enabled,
                artifacts=artifacts,
                device=device,
                swap_roles=True,
                agent_overrides=agent_overrides,
            )
            raw_rows.extend(series)
            summary = summarize_series(series)
            summary_rows.append(summary)
            elapsed = time.perf_counter() - t0
            print(
                "[progress][group-done] "
                f"{group_idx}/{total_groups} game={game_name} pair={a}__vs__{b} "
                f"n_matches={summary.get('n_matches', 0)} "
                f"win_rate_a={summary.get('win_rate_a', 0.0):.3f} "
                f"draw_rate={summary.get('draw_rate', 0.0):.3f} "
                f"elapsed_sec={elapsed:.2f}",
                flush=True,
            )
    return raw_rows, summary_rows


def collect_cross_game_mean(summary_rows: list[dict], group_keys: list[str]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    for row in summary_rows:
        k = tuple(row[g] for g in group_keys)
        buckets.setdefault(k, []).append(row)

    out = []
    for k, rows in buckets.items():
        merged = {group_keys[i]: k[i] for i in range(len(group_keys))}
        merged["n_games"] = len(rows)
        for metric in [
            "win_rate_a",
            "draw_rate",
            "avg_score_a",
            "avg_score_b",
            "avg_game_length",
            "avg_decision_sec_a",
            "avg_decision_sec_b",
            "avg_eval_calls_total_a",
            "avg_eval_calls_total_b",
            "avg_eval_calls_neural_a",
            "avg_eval_calls_neural_b",
            "avg_eval_calls_fallback_a",
            "avg_eval_calls_fallback_b",
            "avg_eval_calls_mixed_a",
            "avg_eval_calls_mixed_b",
            "legal_cache_hit_rate",
            "next_cache_hit_rate",
        ]:
            merged[f"mean_{metric}"] = statistics.fmean(float(r[metric]) for r in rows)
        out.append(merged)
    return out
