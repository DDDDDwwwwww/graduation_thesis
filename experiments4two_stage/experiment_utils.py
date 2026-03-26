from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _runtime_imports():
    from agents import (  # noqa: WPS433
        HeuristicMCTSAgent,
        NeuralValueMCTSAgent,
        PureMCTAgent,
        RandomAgent,
        TwoStageNeuralMCTSAgent,
    )
    from ggp_statemachine import GameStateMachine  # noqa: WPS433

    return {
        "HeuristicMCTSAgent": HeuristicMCTSAgent,
        "NeuralValueMCTSAgent": NeuralValueMCTSAgent,
        "PureMCTAgent": PureMCTAgent,
        "RandomAgent": RandomAgent,
        "TwoStageNeuralMCTSAgent": TwoStageNeuralMCTSAgent,
        "GameStateMachine": GameStateMachine,
    }


@dataclass(frozen=True)
class ValueArtifact:
    model_path: str
    encoder_config_path: str
    vocab_path: str | None = None


@dataclass(frozen=True)
class TwoStageArtifact:
    fast_model_path: str
    fast_encoder_config_path: str
    slow_model_path: str
    slow_encoder_config_path: str
    fast_vocab_path: str | None = None
    slow_vocab_path: str | None = None


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


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
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def init_output_layout(exp_name: str, out_dir: str | Path, args: argparse.Namespace | None = None) -> dict[str, Path]:
    root = ensure_dir(out_dir)
    layout = {
        "root": root,
        "meta": ensure_dir(root / "meta"),
        "raw": ensure_dir(root / "raw"),
        "summary": ensure_dir(root / "summary"),
        "metrics": ensure_dir(root / "metrics"),
        "artifacts": ensure_dir(root / "artifacts"),
    }
    payload = {"experiment": exp_name, "root": str(root)}
    if args is not None:
        payload["args"] = vars(args)
    write_json(layout["meta"] / "run_manifest.json", payload)
    return layout


def default_games(multi: bool = True) -> list[str]:
    if multi:
        return [
            "games/ticTacToe.kif",
            "games/connectFour.kif",
            "games/breakthrough.kif",
        ]
    return ["games/breakthrough.kif"]


def _resolve_path(path_like: str) -> str:
    p = Path(path_like)
    if p.is_absolute():
        return str(p)
    if p.exists():
        return str(p.resolve())
    p2 = ROOT / path_like
    return str(p2.resolve())


def load_artifacts(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {
        "token_mlp": ValueArtifact(
            model_path=_resolve_path(payload["token_mlp"]["model_path"]),
            encoder_config_path=_resolve_path(payload["token_mlp"]["encoder_config_path"]),
            vocab_path=payload["token_mlp"].get("vocab_path"),
        ),
        "token_transformer": ValueArtifact(
            model_path=_resolve_path(payload["token_transformer"]["model_path"]),
            encoder_config_path=_resolve_path(payload["token_transformer"]["encoder_config_path"]),
            vocab_path=payload["token_transformer"].get("vocab_path"),
        ),
        "two_stage_neural_mcts": TwoStageArtifact(
            fast_model_path=_resolve_path(payload["two_stage_neural_mcts"]["fast_model_path"]),
            fast_encoder_config_path=_resolve_path(payload["two_stage_neural_mcts"]["fast_encoder_config_path"]),
            slow_model_path=_resolve_path(payload["two_stage_neural_mcts"]["slow_model_path"]),
            slow_encoder_config_path=_resolve_path(payload["two_stage_neural_mcts"]["slow_encoder_config_path"]),
            fast_vocab_path=payload["two_stage_neural_mcts"].get("fast_vocab_path"),
            slow_vocab_path=payload["two_stage_neural_mcts"].get("slow_vocab_path"),
        ),
    }
    return out


def _build_agent(
    agent_key: str,
    role: str,
    seed: int,
    artifacts: dict,
    iterations: int,
    device: str,
    two_stage_kwargs: dict | None = None,
):
    runtime = _runtime_imports()
    if agent_key == "random":
        return runtime["RandomAgent"](name=f"random_{role}", role=role, seed=seed)
    if agent_key == "pure_mct":
        return runtime["PureMCTAgent"](name=f"pure_{role}", role=role, iterations=iterations, seed=seed)
    if agent_key == "heuristic_mcts":
        return runtime["HeuristicMCTSAgent"](name=f"heur_{role}", role=role, iterations=iterations, seed=seed)
    if agent_key.startswith("neural_mcts:"):
        cfg = agent_key.split(":", 1)[1]
        art = artifacts[cfg]
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
            evaluator_mode="value",
            seed=seed,
        )
    if agent_key == "two_stage_neural_mcts":
        art = artifacts["two_stage_neural_mcts"]
        kw = dict(two_stage_kwargs or {})
        return runtime["TwoStageNeuralMCTSAgent"].from_artifacts(
            name=f"two_stage_{role}",
            role=role,
            fast_model_path=art.fast_model_path,
            fast_encoder_config_path=art.fast_encoder_config_path,
            slow_model_path=art.slow_model_path,
            slow_encoder_config_path=art.slow_encoder_config_path,
            fast_vocab_path=art.fast_vocab_path,
            slow_vocab_path=art.slow_vocab_path,
            iterations=iterations,
            exploration_c=1.4,
            discount_factor=0.99,
            device=device,
            uncertainty_type=kw.get("uncertainty_type", "variance_head"),
            gate_type=kw.get("gate_type", "combined"),
            tau=float(kw.get("tau", 0.15)),
            visit_threshold=int(kw.get("visit_threshold", 4)),
            slow_budget_per_move=int(kw.get("slow_budget_per_move", 16)),
            seed=seed,
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
    artifacts: dict,
    device: str,
    two_stage_kwargs: dict | None = None,
) -> dict:
    runtime = _runtime_imports()
    game = runtime["GameStateMachine"](game_file, cache_enabled=True)
    all_roles = [str(r) for r in game.get_roles()]
    init_state = game.get_initial_state()
    roles = []
    for r in all_roles:
        if r in roles:
            continue
        if game.get_legal_moves(init_state, r):
            roles.append(r)
    if len(roles) != 2:
        raise ValueError(f"Only 2-player games are supported. game={game_file}, roles={roles}")

    role_a = roles[0] if a_is_first_role else roles[1]
    role_b = roles[1] if a_is_first_role else roles[0]
    agent_a = _build_agent(agent_a_key, role_a, seed + 11, artifacts, iterations, device, two_stage_kwargs=two_stage_kwargs)
    agent_b = _build_agent(agent_b_key, role_b, seed + 29, artifacts, iterations, device, two_stage_kwargs=two_stage_kwargs)
    agents = {role_a: agent_a, role_b: agent_b}

    state = init_state
    ply = 0
    decision_seconds = {role_a: [], role_b: []}
    diags = {role_a: [], role_b: []}
    game.clear_caches()
    game.reset_perf_stats()

    while not game.is_terminal(state):
        joint = {}
        for role in roles:
            legal = game.get_legal_moves(state, role)
            t0 = time.perf_counter()
            move = agents[role].select_action(game, state, legal, time_limit=playclock)
            decision_seconds[role].append(time.perf_counter() - t0)
            if hasattr(agents[role], "consume_last_decision_diagnostics"):
                d = agents[role].consume_last_decision_diagnostics()
                if d:
                    diags[role].append(d)
            joint[role] = move
        state = game.get_next_state(state, joint)
        ply += 1

    scores = {r: float(game.get_goal(state, r)) for r in roles}
    perf = game.get_perf_stats()
    max_score = max(scores.values())
    winners = [r for r, s in scores.items() if s == max_score]

    def _diag_summary(role):
        rows = diags[role]
        if not rows:
            return {
                "fast_calls": 0,
                "slow_calls": 0,
                "slow_call_ratio": 0.0,
                "avg_fast_time_sec": 0.0,
                "avg_slow_time_sec": 0.0,
                "leaf_evaluations": 0,
            }
        fast_calls = sum(int(r.get("fast_calls", 0)) for r in rows)
        slow_calls = sum(int(r.get("slow_calls", 0)) for r in rows)
        leaf_evals = sum(int(r.get("leaf_evaluations", 0)) for r in rows)
        fast_time = sum(float(r.get("fast_time_sec", 0.0)) for r in rows)
        slow_time = sum(float(r.get("slow_time_sec", 0.0)) for r in rows)
        return {
            "fast_calls": fast_calls,
            "slow_calls": slow_calls,
            "slow_call_ratio": (slow_calls / max(1, fast_calls)),
            "avg_fast_time_sec": (fast_time / max(1, leaf_evals)),
            "avg_slow_time_sec": (slow_time / max(1, slow_calls)),
            "leaf_evaluations": leaf_evals,
        }

    diag_a = _diag_summary(role_a)
    diag_b = _diag_summary(role_b)
    return {
        "game": Path(game_file).stem,
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
        "legal_calls": int(perf["legal_calls"]),
        "legal_cache_hits": int(perf["legal_cache_hits"]),
        "next_calls": int(perf["next_calls"]),
        "next_cache_hits": int(perf["next_cache_hits"]),
        "fast_calls_a": diag_a["fast_calls"],
        "slow_calls_a": diag_a["slow_calls"],
        "slow_call_ratio_a": diag_a["slow_call_ratio"],
        "avg_fast_eval_time_sec_a": diag_a["avg_fast_time_sec"],
        "avg_slow_eval_time_sec_a": diag_a["avg_slow_time_sec"],
        "leaf_evaluations_a": diag_a["leaf_evaluations"],
        "fast_calls_b": diag_b["fast_calls"],
        "slow_calls_b": diag_b["slow_calls"],
        "slow_call_ratio_b": diag_b["slow_call_ratio"],
        "avg_fast_eval_time_sec_b": diag_b["avg_fast_time_sec"],
        "avg_slow_eval_time_sec_b": diag_b["avg_slow_time_sec"],
        "leaf_evaluations_b": diag_b["leaf_evaluations"],
    }


def run_series(
    game_file: str,
    agent_a_key: str,
    agent_b_key: str,
    rounds: int,
    playclock: float,
    iterations: int,
    seed: int,
    artifacts: dict,
    device: str,
    two_stage_kwargs: dict | None = None,
) -> list[dict]:
    rows = []
    for i in range(int(rounds)):
        first = (i % 2 == 0)
        row = run_single_match(
            game_file=game_file,
            agent_a_key=agent_a_key,
            agent_b_key=agent_b_key,
            a_is_first_role=first,
            playclock=playclock,
            iterations=iterations,
            seed=seed + i * 101,
            artifacts=artifacts,
            device=device,
            two_stage_kwargs=two_stage_kwargs,
        )
        rows.append(row)
        print(
            "[progress][match] "
            f"game={Path(game_file).stem} pair={agent_a_key}__vs__{agent_b_key} "
            f"round={i + 1}/{int(rounds)} winner={row['winner']} "
            f"score_a={row['score_a']:.1f} score_b={row['score_b']:.1f}",
            flush=True,
        )
    return rows


def summarize_series(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    wins = sum(1 for r in rows if r["winner"] == "a")
    losses = sum(1 for r in rows if r["winner"] == "b")
    draws = sum(1 for r in rows if r["winner"] == "draw")
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
        "legal_cache_hit_rate": (legal_hits / legal_calls) if legal_calls else 0.0,
        "next_cache_hit_rate": (next_hits / next_calls) if next_calls else 0.0,
        "mean_fast_calls_a": statistics.fmean(float(r["fast_calls_a"]) for r in rows),
        "mean_slow_calls_a": statistics.fmean(float(r["slow_calls_a"]) for r in rows),
        "mean_slow_call_ratio_a": statistics.fmean(float(r["slow_call_ratio_a"]) for r in rows),
        "mean_fast_eval_time_a": statistics.fmean(float(r["avg_fast_eval_time_sec_a"]) for r in rows),
        "mean_slow_eval_time_a": statistics.fmean(float(r["avg_slow_eval_time_sec_a"]) for r in rows),
        "mean_leaf_eval_a": statistics.fmean(float(r["leaf_evaluations_a"]) for r in rows),
        "mean_fast_calls_b": statistics.fmean(float(r["fast_calls_b"]) for r in rows),
        "mean_slow_calls_b": statistics.fmean(float(r["slow_calls_b"]) for r in rows),
        "mean_slow_call_ratio_b": statistics.fmean(float(r["slow_call_ratio_b"]) for r in rows),
        "mean_fast_eval_time_b": statistics.fmean(float(r["avg_fast_eval_time_sec_b"]) for r in rows),
        "mean_slow_eval_time_b": statistics.fmean(float(r["avg_slow_eval_time_sec_b"]) for r in rows),
        "mean_leaf_eval_b": statistics.fmean(float(r["leaf_evaluations_b"]) for r in rows),
    }


def run_match_grid(
    games: list[str],
    pairs: list[tuple[str, str]],
    rounds: int,
    playclock: float,
    iterations: int,
    seed: int,
    artifacts: dict,
    device: str,
    two_stage_kwargs: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    raw_rows = []
    summary_rows = []
    total = len(games) * len(pairs)
    idx = 0
    for game in games:
        for a, b in pairs:
            idx += 1
            print(f"[progress][group-start] {idx}/{total} game={Path(game).stem} pair={a}__vs__{b}", flush=True)
            series = run_series(
                game_file=game,
                agent_a_key=a,
                agent_b_key=b,
                rounds=rounds,
                playclock=playclock,
                iterations=iterations,
                seed=seed,
                artifacts=artifacts,
                device=device,
                two_stage_kwargs=two_stage_kwargs,
            )
            raw_rows.extend(series)
            summary_rows.append(summarize_series(series))
    return raw_rows, summary_rows


def collect_cross_game_mean(summary_rows: list[dict], group_keys: list[str]) -> list[dict]:
    buckets = {}
    for row in summary_rows:
        key = tuple(row[k] for k in group_keys)
        buckets.setdefault(key, []).append(row)
    out = []
    metric_keys = [
        "win_rate_a",
        "draw_rate",
        "avg_score_a",
        "avg_score_b",
        "avg_game_length",
        "avg_decision_sec_a",
        "avg_decision_sec_b",
        "legal_cache_hit_rate",
        "next_cache_hit_rate",
        "mean_fast_calls_a",
        "mean_slow_calls_a",
        "mean_slow_call_ratio_a",
        "mean_fast_eval_time_a",
        "mean_slow_eval_time_a",
        "mean_leaf_eval_a",
    ]
    for key, rows in buckets.items():
        merged = {group_keys[i]: key[i] for i in range(len(group_keys))}
        merged["n_games"] = len(rows)
        for mk in metric_keys:
            merged[f"mean_{mk}"] = statistics.fmean(float(r.get(mk, 0.0)) for r in rows)
        out.append(merged)
    return out
