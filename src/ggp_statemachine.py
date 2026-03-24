"""GGP 状态机（Prolog 后端）。

功能概览：
1. 读取 GDL/KIF 规则并翻译到 Prolog；
2. 提供通用状态机接口：角色、初始状态、合法动作、状态转移、终局判断、得分；
3. 对 `legal` 与 `next` 查询做缓存，并记录性能计数。
"""

from __future__ import annotations

import re

from pyswip import Prolog

from gdl_parser import GDLTranslator, SExpressionParser


class GameStateMachine:
    """面向智能体的统一游戏状态机封装。"""

    def __init__(self, rule_file, cache_enabled=True):
        """初始化 Prolog 引擎并加载规则。"""
        self.prolog = Prolog()
        self.translator = GDLTranslator()
        self.parser = SExpressionParser()
        self.cache_enabled = bool(cache_enabled)

        self._legal_cache = {}
        self._next_cache = {}
        self._perf_stats = {}
        self.reset_perf_stats()

        # pyswip uses a shared Prolog runtime in-process. Without cleanup, rules
        # from previously loaded games can leak into later matches.
        self._reset_knowledge_base()
        self._load_and_transform_rules(rule_file)

    @staticmethod
    def _prolog_atom(atom) -> str:
        text = str(atom)
        if re.fullmatch(r"[a-z][A-Za-z0-9_]*", text):
            return text
        return "'" + text.replace("'", "''") + "'"

    def _reset_knowledge_base(self) -> None:
        """Clear user-defined dynamic predicates to avoid cross-game contamination."""
        try:
            rows = list(
                self.prolog.query(
                    "predicate_property(H, dynamic), "
                    "\\+ predicate_property(H, imported_from(_)), "
                    "functor(H, N, A)"
                )
            )
        except Exception:
            rows = []

        targets = {(str(r["N"]), int(r["A"])) for r in rows}
        for name, arity in targets:
            if name.startswith("$"):
                continue
            try:
                atom = self._prolog_atom(name)
                list(self.prolog.query(f"abolish({atom}/{int(arity)})"))
            except Exception:
                # Best-effort cleanup; keep going to avoid blocking initialization.
                continue

    def _load_and_transform_rules(self, filename):
        """读取 GDL 文件，解析并断言到 Prolog 知识库。"""
        with open(filename, "r", encoding="utf-8") as f:
            gdl_content = f.read()

        parsed_expressions = self.parser.parse(gdl_content)
        prolog_code = self.translator.translate(parsed_expressions)

        for rule in prolog_code.split("\n"):
            if not rule.strip():
                continue
            try:
                # `assertz` 会自动补句号，因此去掉末尾 '.'。
                self.prolog.assertz(rule[:-1])
            except Exception as e:
                print(f"Prolog Error on line: {rule}\nError: {e}")

    def get_roles(self):
        """查询角色列表。"""
        return [sol["R"] for sol in self.prolog.query("role(R)")]

    def get_initial_state(self):
        """查询初始状态事实列表。"""
        return [sol["F"] for sol in self.prolog.query("init(F)")]

    def get_legal_moves(self, state, role):
        """查询角色合法动作（含缓存与统计）。"""
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
        """计算后继状态（含缓存与统计）。"""
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
        """判断状态是否终局。"""
        self._reset_state(state)
        return list(self.prolog.query("terminal")) != []

    def get_goal(self, state, role):
        """读取角色在状态中的 goal 分值。"""
        self._reset_state(state)
        q = f"goal({role}, V)"
        for sol in self.prolog.query(q):
            return sol["V"]
        return 0

    def _reset_state(self, state):
        """把 Prolog 中 `true/1` 重置为当前状态事实。"""
        self.prolog.retractall("true(_)")
        for fact in state:
            self.prolog.assertz(f"true({fact})")

    def _inject_moves(self, moves):
        """把联合动作写入 `does/2`。"""
        for role, move in moves.items():
            # 防御性处理：None 写入 Prolog 会变成变量，统一改为 noop。
            if move is None or str(move) == "None":
                move = "noop"
            self.prolog.assertz(f"does({role}, {move})")

    def _clean_moves(self):
        """清空当前步动作事实。"""
        self.prolog.retractall("does(_,_)")

    def clear_caches(self):
        """清空缓存。"""
        self._legal_cache.clear()
        self._next_cache.clear()

    def reset_perf_stats(self):
        """重置性能统计。"""
        self._perf_stats = {
            "legal_calls": 0,
            "legal_cache_hits": 0,
            "next_calls": 0,
            "next_cache_hits": 0,
        }

    def get_perf_stats(self):
        """返回性能统计快照。"""
        return dict(self._perf_stats)

    def _state_key(self, state):
        """状态转 key：排序后字符串元组，保证确定性。"""
        return tuple(sorted(str(fact) for fact in state))

    def _moves_key(self, moves):
        """联合动作转 key：按角色排序后元组。"""
        return tuple(sorted((str(role), str(move)) for role, move in moves.items()))

    def get_state_facts_as_strings(self, state):
        """返回排序后的状态事实字符串列表。"""
        return sorted(str(fact) for fact in state)

    def get_current_role(self, state):
        """尝试从 `control/1` 推断当前行动方。"""
        self._reset_state(state)
        try:
            controls = [sol["R"] for sol in self.prolog.query("control(R)")]
            if controls:
                return str(controls[0])
        except Exception:
            pass
        return None

    def extract_board_facts(self, state):
        """提取棋盘类事实（默认识别 `cell(...)`）。"""
        return [fact for fact in self.get_state_facts_as_strings(state) if fact.startswith("cell(")]

    def get_role_index(self, role):
        """返回角色在 `self.roles` 中的索引位置。"""
        role_str = str(role)
        for idx, value in enumerate(self.get_roles()):
            if str(value) == role_str:
                return idx
        raise ValueError(f"Unknown role: {role_str}")
