from __future__ import annotations

"""事实词汇表定义。

用于把符号状态中的 fact 字符串稳定映射为整数索引。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class FactVocabulary:
    """fact 字符串到索引的确定性映射。"""

    fact_to_index: dict[str, int]
    index_to_fact: list[str]
    unknown_token: str = "<UNK>"

    @classmethod
    def empty(cls, unknown_token: str = "<UNK>") -> "FactVocabulary":
        """构造仅含未知词的空词汇表。"""
        return cls(
            fact_to_index={unknown_token: 0},
            index_to_fact=[unknown_token],
            unknown_token=unknown_token,
        )

    @classmethod
    def fit(cls, states_or_fact_lists: Iterable[Iterable[str]], unknown_token: str = "<UNK>") -> "FactVocabulary":
        """从状态事实集合中构建词汇表。

        采用排序后写入，确保不同机器/运行次序下索引一致。
        """
        vocab = cls.empty(unknown_token=unknown_token)
        seen: set[str] = set()
        for facts in states_or_fact_lists:
            for fact in facts:
                seen.add(str(fact))

        for fact in sorted(seen):
            if fact == unknown_token:
                continue
            vocab.fact_to_index[fact] = len(vocab.index_to_fact)
            vocab.index_to_fact.append(fact)
        return vocab

    @property
    def size(self) -> int:
        """词汇总大小（含 `<UNK>`）。"""
        return len(self.index_to_fact)

    def encode_fact(self, fact: str) -> int:
        """将 fact 编码为索引，未登录词回退到 `<UNK>`。"""
        return self.fact_to_index.get(str(fact), 0)

    def save(self, path: str | Path) -> None:
        """将词汇表保存为 JSON。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "unknown_token": self.unknown_token,
            "index_to_fact": self.index_to_fact,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FactVocabulary":
        """从 JSON 加载词汇表。"""
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        index_to_fact = [str(x) for x in payload["index_to_fact"]]
        unknown_token = str(payload.get("unknown_token", "<UNK>"))
        fact_to_index = {fact: idx for idx, fact in enumerate(index_to_fact)}
        if unknown_token not in fact_to_index:
            index_to_fact.insert(0, unknown_token)
            fact_to_index = {fact: idx for idx, fact in enumerate(index_to_fact)}
        return cls(
            fact_to_index=fact_to_index,
            index_to_fact=index_to_fact,
            unknown_token=unknown_token,
        )
