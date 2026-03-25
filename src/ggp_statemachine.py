from __future__ import annotations

import re

from pyswip import Prolog

from gdl_parser import GDLTranslator, SExpressionParser


class GameStateMachine:
    """Generic game state machine backed by SWI-Prolog."""

    _global_loaded_predicates: set[tuple[str, int]] = set()

    def __init__(self, rule_file, cache_enabled=True):
        self.prolog = Prolog()
        self.translator = GDLTranslator()
        self.parser = SExpressionParser()
        self.cache_enabled = bool(cache_enabled)

        self._legal_cache = {}
        self._next_cache = {}
        self._perf_stats = {}
        self.reset_perf_stats()

        # pyswip uses a shared runtime in-process, so clean previously loaded
        # game predicates before loading this game's rules.
        self._reset_knowledge_base()
        self._load_and_transform_rules(rule_file)

    @staticmethod
    def _prolog_atom(atom) -> str:
        text = str(atom)
        if re.fullmatch(r"[a-z][A-Za-z0-9_]*", text):
            return text
        return "'" + text.replace("'", "''") + "'"

    def _reset_knowledge_base(self) -> None:
        """Clear only predicates previously loaded by this project."""
        targets = set(GameStateMachine._global_loaded_predicates)
        targets.update({("true", 1), ("does", 2)})
        for name, arity in sorted(targets):
            if name.startswith("$"):
                continue
            try:
                atom = self._prolog_atom(name)
                list(self.prolog.query(f"abolish({atom}/{int(arity)})"))
            except Exception:
                # Best-effort cleanup; keep going.
                continue
            GameStateMachine._global_loaded_predicates.discard((name, int(arity)))

    @staticmethod
    def _split_top_level_args(text: str) -> list[str]:
        parts: list[str] = []
        buf: list[str] = []
        depth = 0
        for ch in text:
            if ch == "," and depth == 0:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
                continue
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1
            buf.append(ch)
        tail = "".join(buf).strip()
        if tail:
            parts.append(tail)
        return parts

    @classmethod
    def _extract_head_predicate(cls, rule: str) -> tuple[str, int] | None:
        text = rule.strip()
        if not text:
            return None
        if text.endswith("."):
            text = text[:-1]
        head = text.split(":-", 1)[0].strip()
        m = re.match(r"^([a-z][A-Za-z0-9_]*)\s*(?:\((.*)\))?$", head)
        if not m:
            return None
        name = m.group(1)
        args = m.group(2)
        if not args:
            return (name, 0)
        return (name, len(cls._split_top_level_args(args)))

    def _load_and_transform_rules(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            gdl_content = f.read()

        parsed_expressions = self.parser.parse(gdl_content)
        prolog_code = self.translator.translate(parsed_expressions)

        for rule in prolog_code.split("\n"):
            if not rule.strip():
                continue
            pred = self._extract_head_predicate(rule)
            try:
                # assertz auto-handles trailing dot normalization.
                self.prolog.assertz(rule[:-1])
                if pred is not None:
                    GameStateMachine._global_loaded_predicates.add(pred)
            except Exception as e:
                print(f"Prolog Error on line: {rule}\nError: {e}")

    def get_roles(self):
        return [sol["R"] for sol in self.prolog.query("role(R)")]

    def get_initial_state(self):
        return [sol["F"] for sol in self.prolog.query("init(F)")]

    def get_legal_moves(self, state, role):
        self._perf_stats["legal_calls"] += 1

        state_key = self._state_key(state)
        if self.cache_enabled:
            cache_key = (state_key, str(role))
            cached_moves = self._legal_cache.get(cache_key)
            if cached_moves is not None:
                self._perf_stats["legal_cache_hits"] += 1
                return list(cached_moves)

        self._reset_state(state)
        query = f"legal({role}, M)"
        solutions = [sol["M"] for sol in self.prolog.query(query)]
        unique_moves = sorted(set(solutions), key=lambda m: str(m))

        if self.cache_enabled:
            self._legal_cache[(state_key, str(role))] = tuple(unique_moves)
        return unique_moves

    def get_next_state(self, state, moves):
        self._perf_stats["next_calls"] += 1

        state_key = self._state_key(state)
        moves_key = self._moves_key(moves)
        if self.cache_enabled:
            cache_key = (state_key, moves_key)
            cached_state = self._next_cache.get(cache_key)
            if cached_state is not None:
                self._perf_stats["next_cache_hits"] += 1
                return list(cached_state)

        self._reset_state(state)
        self._inject_moves(moves)
        next_state = [sol["F"] for sol in self.prolog.query("next(F)")]
        unique_state = sorted(set(next_state), key=lambda f: str(f))
        self._clean_moves()

        if self.cache_enabled:
            self._next_cache[(state_key, moves_key)] = tuple(unique_state)
        return unique_state

    def is_terminal(self, state):
        self._reset_state(state)
        return list(self.prolog.query("terminal")) != []

    def get_goal(self, state, role):
        self._reset_state(state)
        q = f"goal({role}, V)"
        for sol in self.prolog.query(q):
            return sol["V"]
        return 0

    def _reset_state(self, state):
        self.prolog.retractall("true(_)")
        for fact in state:
            self.prolog.assertz(f"true({fact})")

    def _inject_moves(self, moves):
        for role, move in moves.items():
            # Prevent None from being interpreted as a variable in Prolog.
            if move is None or str(move) == "None":
                move = "noop"
            self.prolog.assertz(f"does({role}, {move})")

    def _clean_moves(self):
        self.prolog.retractall("does(_,_)")

    def clear_caches(self):
        self._legal_cache.clear()
        self._next_cache.clear()

    def reset_perf_stats(self):
        self._perf_stats = {
            "legal_calls": 0,
            "legal_cache_hits": 0,
            "next_calls": 0,
            "next_cache_hits": 0,
        }

    def get_perf_stats(self):
        return dict(self._perf_stats)

    def _state_key(self, state):
        return tuple(sorted(str(fact) for fact in state))

    def _moves_key(self, moves):
        return tuple(sorted((str(role), str(move)) for role, move in moves.items()))

    def get_state_facts_as_strings(self, state):
        return sorted(str(fact) for fact in state)

    def get_current_role(self, state):
        self._reset_state(state)
        try:
            controls = [sol["R"] for sol in self.prolog.query("control(R)")]
            if controls:
                return str(controls[0])
        except Exception:
            pass
        return None

    def extract_board_facts(self, state):
        return [fact for fact in self.get_state_facts_as_strings(state) if fact.startswith("cell(")]

    def get_role_index(self, role):
        role_str = str(role)
        for idx, value in enumerate(self.get_roles()):
            if str(value) == role_str:
                return idx
        raise ValueError(f"Unknown role: {role_str}")
