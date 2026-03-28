from __future__ import annotations

"""Convert legacy JSONL value samples to SDRPV dataset format.

Output fields per sample:
  - s: {"state_facts", "acting_role", "ply_index", "terminal"}
  - b: cheap baseline value in [-1, 1]
  - q_t: teacher search value in [-1, 1]
  - z: clipped terminal supervision in [-1, 1]
  - phi: phase-aware global features
  - metadata: game/match/source and conversion settings
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agents.pure_mct_agent import PureMCTAgent
from ggp_statemachine import GameStateMachine
from nn.inference import load_value_artifacts, predict_value


def clip_unit(x: float) -> float:
    return max(-1.0, min(1.0, float(x)))


def is_finite(x: float) -> bool:
    return math.isfinite(float(x))


def goal_to_unit_value(v: float) -> float:
    """Map common GGP goal scale to [-1, 1].

    Most games use [0,100] goals. For that case use (v-50)/50.
    If value already looks normalized, keep it as-is then clip.
    """
    x = float(v)
    if abs(x) <= 1.0:
        return clip_unit(x)
    return clip_unit((x - 50.0) / 50.0)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    q = max(0.0, min(1.0, float(q)))
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


class JsonlCache:
    """Append-only JSONL key-value cache."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mem: dict[str, float] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        self.mem[str(row["key"])] = float(row["value"])
                    except Exception:
                        continue
        self._append_fp = self.path.open("a", encoding="utf-8")

    def get(self, key: str) -> float | None:
        v = self.mem.get(key)
        if v is None:
            return None
        return float(v)

    def set(self, key: str, value: float) -> None:
        value = float(value)
        self.mem[key] = value
        self._append_fp.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
        self._append_fp.flush()

    def close(self) -> None:
        self._append_fp.close()


def read_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                yield line_no, None


def discover_game_files() -> dict[str, Path]:
    games_dir = ROOT / "games"
    out: dict[str, Path] = {}
    for p in games_dir.glob("*.kif"):
        out[p.stem] = p
    return out


def state_key(game_name: str, acting_role: str, state_facts: list[str]) -> str:
    payload = {
        "game_name": str(game_name),
        "acting_role": str(acting_role),
        "state_facts": sorted(str(x) for x in state_facts),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_phase_index(rows: list[dict]) -> dict[tuple[str, str | int], int]:
    max_ply_by_match: dict[tuple[str, str | int], int] = defaultdict(int)
    for row in rows:
        game_name = str(row.get("game_name", ""))
        match_id = row.get("match_id", -1)
        ply_index = int(row.get("ply_index", 0) or 0)
        k = (game_name, match_id)
        if ply_index > max_ply_by_match[k]:
            max_ply_by_match[k] = ply_index
    return dict(max_ply_by_match)


def phase_of(ply_index: int, max_ply: int) -> str:
    if max_ply <= 0:
        return "midgame"
    r = float(ply_index) / float(max_ply)
    if r <= 0.33:
        return "opening"
    if r <= 0.66:
        return "midgame"
    return "endgame"


def keep_by_phase_sampling(
    enabled: bool,
    row: dict,
    max_ply_by_match: dict[tuple[str, str | int], int],
    counter: dict[tuple[str, str | int, str], int],
) -> bool:
    if not enabled:
        return True
    game_name = str(row.get("game_name", ""))
    match_id = row.get("match_id", -1)
    ply_index = int(row.get("ply_index", 0) or 0)
    max_ply = int(max_ply_by_match.get((game_name, match_id), 0))
    ph = phase_of(ply_index, max_ply)
    quota = {"opening": 2, "midgame": 4, "endgame": 2}.get(ph, 2)
    ck = (game_name, match_id, ph)
    if counter[ck] >= quota:
        return False
    counter[ck] += 1
    return True


def parse_cell_fact(fact: str):
    text = str(fact).strip()
    if not text.startswith("cell(") or not text.endswith(")"):
        return None
    inner = text[5:-1]
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) < 3:
        return None
    return parts[0], parts[1], parts[2]


def extract_phase_features(row: dict, game: GameStateMachine) -> dict:
    facts = [str(x) for x in row["state_facts"]]
    acting_role = str(row["acting_role"])
    ply_index = int(row.get("ply_index", 0) or 0)
    terminal = bool(row.get("terminal", False))

    state = facts
    legal_moves = game.get_legal_moves(state, acting_role)
    legal_move_count = int(len(legal_moves))

    cell_items = [parse_cell_fact(f) for f in facts]
    cell_items = [x for x in cell_items if x is not None]
    occupied = 0
    empties = 0
    piece_count_by_owner: dict[str, int] = defaultdict(int)
    x_values = set()
    y_values = set()
    for x, y, c in cell_items:
        x_values.add(x)
        y_values.add(y)
        c_norm = str(c).lower()
        if c_norm in {"b", "blank", "empty", "none"}:
            empties += 1
            continue
        occupied += 1
        piece_count_by_owner[str(c)] += 1

    board_cells = occupied + empties
    if board_cells <= 0 and x_values and y_values:
        board_cells = len(x_values) * len(y_values)
    occupancy = 0.0 if board_cells <= 0 else float(occupied) / float(board_cells)

    my_pieces = piece_count_by_owner.get(acting_role, 0)
    opp_pieces = 0
    for k, v in piece_count_by_owner.items():
        if k == acting_role:
            continue
        opp_pieces += int(v)
    piece_diff = float(my_pieces - opp_pieces) if piece_count_by_owner else None

    move_progress = math.tanh(max(0.0, float(ply_index)) / 50.0)
    if terminal:
        terminal_proxy = 1.0
    elif legal_move_count <= 1:
        terminal_proxy = 0.8
    else:
        terminal_proxy = max(0.0, min(1.0, 0.5 * move_progress + 0.5 * occupancy))

    return {
        "move_progress": float(move_progress),
        "board_occupancy_ratio": float(max(0.0, min(1.0, occupancy))),
        "piece_count_diff": piece_diff,
        "legal_move_count": legal_move_count,
        "terminal_proximity_proxy": float(max(0.0, min(1.0, terminal_proxy))),
    }


def search_value(
    game: GameStateMachine,
    state_facts: list[str],
    acting_role: str,
    sims: int,
    seed: int,
) -> float:
    agent = PureMCTAgent(
        name=f"eval_{acting_role}",
        role=str(acting_role),
        iterations=max(1, int(sims)),
        seed=int(seed),
    )
    roles = [str(r) for r in game.get_roles()]
    root = agent._prepare_root(state_facts)
    for _ in range(max(1, int(sims))):
        ok = agent._run_single_iteration(game, root, roles, budget_end=None)
        if not ok:
            break
    return goal_to_unit_value(float(root.mean_value))


def build_model_predictor(model_path: str, vocab_path: str | None, encoder_path: str | None, device: str):
    model, encoder, _ = load_value_artifacts(
        model_path=model_path,
        vocab_path=vocab_path,
        encoder_config_path=encoder_path,
        device=device,
    )

    def _predict(row: dict) -> float:
        facts = [str(x) for x in row["state_facts"]]
        role = str(row["acting_role"])
        ply_index = int(row.get("ply_index", 0) or 0)
        terminal = bool(row.get("terminal", False))
        if hasattr(encoder, "encode_facts"):
            x = encoder.encode_facts(facts, role=role, ply_index=ply_index, terminal=terminal)
        else:
            x = encoder.encode(facts, game=None, role=role, ply_index=ply_index, terminal=terminal)
        return clip_unit(float(predict_value(model, x, device=device)))

    return _predict


_WORKER_ENGINES: dict[str, GameStateMachine] = {}
_WORKER_GAME_FILES: dict[str, str] = {}
_WORKER_BASELINE_MODE = "shallow_mcts"
_WORKER_BASELINE_SIMS = 64
_WORKER_TEACHER_SIMS = 600
_WORKER_SEED = 42
_WORKER_PREDICT_BASE = None


def worker_init(
    game_files: dict[str, str],
    baseline_mode: str,
    baseline_sims: int,
    teacher_sims: int,
    seed: int,
    base_model_path: str | None,
    base_vocab_path: str | None,
    base_encoder_path: str | None,
    device: str,
) -> None:
    global _WORKER_GAME_FILES
    global _WORKER_BASELINE_MODE
    global _WORKER_BASELINE_SIMS
    global _WORKER_TEACHER_SIMS
    global _WORKER_SEED
    global _WORKER_PREDICT_BASE
    _WORKER_GAME_FILES = dict(game_files)
    _WORKER_BASELINE_MODE = str(baseline_mode)
    _WORKER_BASELINE_SIMS = int(baseline_sims)
    _WORKER_TEACHER_SIMS = int(teacher_sims)
    _WORKER_SEED = int(seed)
    if _WORKER_BASELINE_MODE == "base_net":
        _WORKER_PREDICT_BASE = build_model_predictor(
            model_path=str(base_model_path),
            vocab_path=base_vocab_path,
            encoder_path=base_encoder_path,
            device=device,
        )
    else:
        _WORKER_PREDICT_BASE = None


def _worker_get_game(game_name: str) -> GameStateMachine:
    game = _WORKER_ENGINES.get(game_name)
    if game is not None:
        return game
    game_file = _WORKER_GAME_FILES.get(game_name)
    if game_file is None:
        raise ValueError(f"Game not found for worker: {game_name}")
    game = GameStateMachine(game_file)
    _WORKER_ENGINES[game_name] = game
    return game


def worker_process_task(task: dict) -> dict:
    try:
        idx = int(task["idx"])
        row = dict(task["row"])
        key = str(task["key"])
        z = float(task["z"])
        game_name = str(row.get("game_name", ""))
        acting_role = str(row["acting_role"])
        facts = [str(x) for x in row["state_facts"]]
        base_cache_key = str(task["base_cache_key"])
        teacher_cache_key = str(task["teacher_cache_key"])

        game = _worker_get_game(game_name)
        b = task.get("b")
        if b is None:
            if _WORKER_BASELINE_MODE == "shallow_mcts":
                b = search_value(
                    game=game,
                    state_facts=facts,
                    acting_role=acting_role,
                    sims=_WORKER_BASELINE_SIMS,
                    seed=int(_WORKER_SEED + idx * 13),
                )
            else:
                b = float(_WORKER_PREDICT_BASE(row))
        q_t = task.get("q_t")
        if q_t is None:
            q_t = search_value(
                game=game,
                state_facts=facts,
                acting_role=acting_role,
                sims=_WORKER_TEACHER_SIMS,
                seed=int(_WORKER_SEED + idx * 29),
            )

        b = clip_unit(float(b))
        q_t = clip_unit(float(q_t))
        if (not is_finite(b)) or (not is_finite(q_t)):
            return {"ok": False, "idx": idx, "kind": "nan_or_inf"}

        phi = extract_phase_features(row, game)
        sample = {
            "_key": key,
            "s": {
                "state_facts": facts,
                "acting_role": acting_role,
                "ply_index": int(row.get("ply_index", 0) or 0),
                "terminal": bool(row.get("terminal", False)),
            },
            "b": b,
            "q_t": q_t,
            "z": z,
            "phi": phi,
            "game_name": game_name,
            "match_id": row.get("match_id"),
            "ply_index": int(row.get("ply_index", 0) or 0),
            "source_agent": row.get("source_agent"),
            "baseline_mode": _WORKER_BASELINE_MODE,
            "teacher_sims": int(_WORKER_TEACHER_SIMS),
            "student_sims": int(task["student_sims"]),
        }
        return {
            "ok": True,
            "idx": idx,
            "sample": sample,
            "b": b,
            "q_t": q_t,
            "base_cache_key": base_cache_key,
            "teacher_cache_key": teacher_cache_key,
        }
    except Exception as exc:
        return {"ok": False, "idx": int(task.get("idx", -1)), "kind": "exception", "error": str(exc)}


def load_input_rows(paths: list[Path], game_filter: str | None, max_samples: int | None) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        for _, row in read_jsonl(path):
            if row is None:
                continue
            if game_filter and str(row.get("game_name", "")) != str(game_filter):
                continue
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                return rows
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert legacy datasets to SDRPV JSONL.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or multiple legacy JSONL files.",
    )
    parser.add_argument("--output", required=True, help="Output SDRPV JSONL file path.")
    parser.add_argument("--game", default=None, help="Optional game_name filter.")
    parser.add_argument("--baseline-mode", choices=["shallow_mcts", "base_net"], default="shallow_mcts")
    parser.add_argument("--teacher-sims", type=int, default=600)
    parser.add_argument("--student-sims", type=int, default=120)
    parser.add_argument("--baseline-sims", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--phase-sampling", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-cache", default="outputs/cache/sdrpv_baseline_cache.jsonl")
    parser.add_argument("--teacher-cache", default="outputs/cache/sdrpv_teacher_cache.jsonl")
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--base-vocab-path", default=None)
    parser.add_argument("--base-encoder-path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel worker count for labeling.")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N processed rows.")
    parser.add_argument(
        "--heartbeat-sec",
        type=int,
        default=60,
        help="Print heartbeat log every N seconds even without completed rows.",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input dataset not found: {p}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_cache = JsonlCache(Path(args.baseline_cache))
    teacher_cache = JsonlCache(Path(args.teacher_cache))
    try:
        rows = load_input_rows(input_paths, game_filter=args.game, max_samples=args.max_samples)
        if not rows:
            raise ValueError("No rows selected from input datasets.")

        max_ply_by_match = build_phase_index(rows)
        game_files = discover_game_files()

        predict_base = None
        if args.baseline_mode == "base_net":
            if not args.base_model_path or not args.base_encoder_path:
                raise ValueError(
                    "baseline-mode=base_net requires --base-model-path and --base-encoder-path."
                )
            predict_base = build_model_predictor(
                model_path=args.base_model_path,
                vocab_path=args.base_vocab_path,
                encoder_path=args.base_encoder_path,
                device=args.device,
            )

        done_keys = set()
        if args.resume and output_path.exists():
            for _, prev in read_jsonl(output_path):
                if prev is None:
                    continue
                key = prev.get("_key")
                if key:
                    done_keys.add(str(key))

        rows_total = len(rows)
        stats = {
            "total_input": rows_total,
            "success": 0,
            "skip": 0,
            "fail": 0,
            "missing_fields": 0,
            "nan_or_inf": 0,
            "missing_game": 0,
        }
        phase_counter: dict[tuple[str, str | int, str], int] = defaultdict(int)
        b_values: list[float] = []
        q_values: list[float] = []
        z_values: list[float] = []
        residual_abs: list[float] = []
        tasks: list[dict] = []
        prepared_keys: set[str] = set()

        print(
            f"[convert_sdrpv] preparing tasks... rows={rows_total}, "
            f"num_workers={max(1, int(args.num_workers))}"
        )
        for idx, row in enumerate(rows, 1):
            if not keep_by_phase_sampling(
                enabled=bool(args.phase_sampling),
                row=row,
                max_ply_by_match=max_ply_by_match,
                counter=phase_counter,
            ):
                stats["skip"] += 1
                continue

            needed = ["state_facts", "acting_role", "value_target"]
            if any(k not in row for k in needed):
                stats["missing_fields"] += 1
                stats["skip"] += 1
                continue

            game_name = str(row.get("game_name", ""))
            if not game_name:
                stats["missing_game"] += 1
                stats["skip"] += 1
                continue
            if game_name not in game_files:
                stats["missing_game"] += 1
                stats["skip"] += 1
                continue

            facts = [str(x) for x in row["state_facts"]]
            acting_role = str(row["acting_role"])
            key = state_key(game_name, acting_role, facts)
            if key in done_keys or key in prepared_keys:
                stats["skip"] += 1
                continue

            z = clip_unit(float(row["value_target"]))
            if not is_finite(z):
                stats["nan_or_inf"] += 1
                stats["skip"] += 1
                continue

            base_cache_key = f"{args.baseline_mode}:{game_name}:{acting_role}:{key}"
            teacher_cache_key = f"teacher:{args.teacher_sims}:{game_name}:{acting_role}:{key}"
            b = baseline_cache.get(base_cache_key)
            q_t = teacher_cache.get(teacher_cache_key)
            tasks.append(
                {
                    "idx": idx,
                    "row": {
                        "state_facts": facts,
                        "acting_role": acting_role,
                        "ply_index": int(row.get("ply_index", 0) or 0),
                        "terminal": bool(row.get("terminal", False)),
                        "game_name": game_name,
                        "match_id": row.get("match_id"),
                        "source_agent": row.get("source_agent"),
                    },
                    "key": key,
                    "z": z,
                    "b": b,
                    "q_t": q_t,
                    "base_cache_key": base_cache_key,
                    "teacher_cache_key": teacher_cache_key,
                    "student_sims": int(args.student_sims),
                }
            )
            prepared_keys.add(key)
            if idx % max(1, int(args.log_every)) == 0:
                print(f"[convert_sdrpv] prepare_progress={idx}/{rows_total}, tasks={len(tasks)}")

        print(f"[convert_sdrpv] task_ready={len(tasks)}")

        out_mode = "a" if args.resume else "w"
        with output_path.open(out_mode, encoding="utf-8") as fout:
            processed = 0
            t0 = time.time()
            last_heartbeat = t0

            def _consume_result(res: dict) -> None:
                nonlocal processed, last_heartbeat
                processed += 1
                if not res.get("ok"):
                    if res.get("kind") == "nan_or_inf":
                        stats["nan_or_inf"] += 1
                    stats["fail"] += 1
                else:
                    sample = res["sample"]
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    b = float(res["b"])
                    q_t = float(res["q_t"])
                    z = float(sample["z"])
                    b_values.append(b)
                    q_values.append(q_t)
                    z_values.append(z)
                    residual_abs.append(abs(q_t - b))
                    stats["success"] += 1
                    done_keys.add(str(sample["_key"]))
                    baseline_cache.set(str(res["base_cache_key"]), b)
                    teacher_cache.set(str(res["teacher_cache_key"]), q_t)

                now = time.time()
                should_print = (
                    processed % max(1, int(args.log_every)) == 0
                    or now - last_heartbeat >= max(1, int(args.heartbeat_sec))
                    or processed == len(tasks)
                )
                if should_print:
                    elapsed = max(1e-6, now - t0)
                    rate = processed / elapsed
                    print(
                        "[convert_sdrpv] "
                        f"processed={processed}/{len(tasks)} "
                        f"success={stats['success']} skip={stats['skip']} fail={stats['fail']} "
                        f"rate={rate:.2f}/s"
                    )
                    last_heartbeat = now

            if int(args.num_workers) <= 1:
                if args.baseline_mode == "base_net":
                    predict_base = build_model_predictor(
                        model_path=args.base_model_path,
                        vocab_path=args.base_vocab_path,
                        encoder_path=args.base_encoder_path,
                        device=args.device,
                    )
                    # expose predictor to single-process worker path
                    worker_init(
                        game_files={k: str(v) for k, v in game_files.items()},
                        baseline_mode=args.baseline_mode,
                        baseline_sims=int(args.baseline_sims),
                        teacher_sims=int(args.teacher_sims),
                        seed=int(args.seed),
                        base_model_path=args.base_model_path,
                        base_vocab_path=args.base_vocab_path,
                        base_encoder_path=args.base_encoder_path,
                        device=args.device,
                    )
                else:
                    worker_init(
                        game_files={k: str(v) for k, v in game_files.items()},
                        baseline_mode=args.baseline_mode,
                        baseline_sims=int(args.baseline_sims),
                        teacher_sims=int(args.teacher_sims),
                        seed=int(args.seed),
                        base_model_path=None,
                        base_vocab_path=None,
                        base_encoder_path=None,
                        device=args.device,
                    )
                for task in tasks:
                    _consume_result(worker_process_task(task))
            else:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=max(1, int(args.num_workers)),
                    initializer=worker_init,
                    initargs=(
                        {k: str(v) for k, v in game_files.items()},
                        args.baseline_mode,
                        int(args.baseline_sims),
                        int(args.teacher_sims),
                        int(args.seed),
                        args.base_model_path,
                        args.base_vocab_path,
                        args.base_encoder_path,
                        args.device,
                    ),
                ) as ex:
                    futures = [ex.submit(worker_process_task, t) for t in tasks]
                    for fut in concurrent.futures.as_completed(futures):
                        _consume_result(fut.result())

        def metric_pack(values: list[float]) -> dict:
            if not values:
                return {"count": 0, "mean": 0.0, "std": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
            return {
                "count": len(values),
                "mean": float(mean(values)),
                "std": float(pstdev(values)),
                "p05": float(quantile(values, 0.05)),
                "p50": float(quantile(values, 0.50)),
                "p95": float(quantile(values, 0.95)),
            }

        summary = {
            "args": vars(args),
            "stats": stats,
            "success_rate": (float(stats["success"]) / float(max(1, rows_total))),
            "b": metric_pack(b_values),
            "q_t": metric_pack(q_values),
            "z": metric_pack(z_values),
            "|q_t-b|": metric_pack(residual_abs),
        }
        if summary["q_t"]["std"] <= 1e-12 or summary["b"]["std"] <= 1e-12:
            print(
                "[convert_sdrpv][WARN] detected near-constant q_t or b. "
                "Please check search value mapping / teacher budget."
            )
        stats_path = output_path.with_suffix(output_path.suffix + ".stats.json")
        stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[convert_sdrpv] done. output={output_path}")
        print(f"[convert_sdrpv] stats={stats_path}")
        print(
            "[convert_sdrpv] "
            f"success={stats['success']}, skip={stats['skip']}, fail={stats['fail']}, "
            f"success_rate={summary['success_rate']:.4f}"
        )
    finally:
        baseline_cache.close()
        teacher_cache.close()


if __name__ == "__main__":
    main()
