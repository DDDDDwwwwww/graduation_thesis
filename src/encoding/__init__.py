"""状态编码模块导出入口。"""

from .fact_vector_encoder import FactVectorEncoder
from .vocab import FactVocabulary

__all__ = ["FactVocabulary", "FactVectorEncoder"]
