from pyswip import Prolog
from gdl_parser import SExpressionParser, GDLTranslator


class GameStateMachine:
    def __init__(self, rule_file, cache_enabled=True):
        self.prolog = Prolog()
        self.translator = GDLTranslator()
        self.parser = SExpressionParser()
        # 统一缓存开关：用于 A/B 对照时显式关闭状态机缓存。
        self.cache_enabled = bool(cache_enabled)

        self._legal_cache = {}
        self._next_cache = {}
        self._perf_stats = {}
        self.reset_perf_stats()

        self._load_and_transform_rules(rule_file)

    def _load_and_transform_rules(self, filename):
        # 规则文件固定按 UTF-8 读取，避免 Windows 默认编码导致解析失败。
        with open(filename, "r", encoding="utf-8") as f:
            gdl_content = f.read()

        parsed_expressions = self.parser.parse(gdl_content)
        prolog_code = self.translator.translate(parsed_expressions)

        for rule in prolog_code.split("\n"):
            if rule.strip():
                try:
                    # Drop trailing '.' because assertz appends it automatically.
                    self.prolog.assertz(rule[:-1])
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
        unique_moves = list(set(solutions))

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
        unique_state = list(set(next_state))
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
        res = list(self.prolog.query(q))
        if res:
            return res[0]["V"]
        return 0

    def _reset_state(self, state):
        self.prolog.retractall("true(_)")
        for fact in state:
            self.prolog.assertz(f"true({fact})")

    def _inject_moves(self, moves):
        for role, move in moves.items():
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
