# GGP 机器人实现规格说明（面向 Codex）

## 1. 目标

在现有代码库基础上实现一个**通用游戏博弈 (GGP) 机器人流水线**：

- `game_runner.py`
- `gdl_parser.py`
- `ggp_statemachine.py`
- `ggp_agent.py`

目标系统应支持：

1. 用于实验的稳定**搜索基线**
2. 通过自我对弈/搜索对弈进行离线**数据集生成**
3. 用于神经网络训练的**状态编码**
4. 用于状态评估的**价值网络**
5. 一个采用**神经价值引导的 MCTS 智能体**，其风格类似于“Fast and Knowledge-Free Deep Learning for GGP”论文的方法
6. 一个用于比较所有必需智能体的完整**实验流水线**

本规格说明旨在作为一份可由 Codex 直接执行的工程计划。

---

## 2. 高层策略

**不要**重写整个项目。

保持当前以以下模块为中心的架构：

- `GameStateMachine` 作为统一的游戏接口
- `GameRunner` 作为比赛/基准测试驱动程序
- 现有的 MCTS 逻辑作为所有搜索智能体的基础

按层构建新功能：

### 阶段 A：稳定并模块化现有搜索智能体

必需的智能体：
- `RandomAgent`
- `PureMCTAgent`
- `HeuristicMCTSAgent`

### 阶段 B：添加神经网络流水线

必需的新模块：
- 状态编码器
- 离线数据集生成
- 价值网络
- `ValueGreedyAgent`
- `NeuralValueMCTSAgent`

### 阶段 C：构建实验

必需的脚本：
- 自我对弈数据集生成
- 训练
- 评估比赛
- 消融实验
- 基准测试摘要

---

## 3. 最终必需的智能体集合

最终实现必须支持以下智能体。

### 3.1 `RandomAgent`
已存在。保留作为最低基线。

### 3.2 `PureMCTAgent`
概念上已存在。这必须保持为标准的**纯 UCT + 随机 rollout** 基线。

定义：
- 树策略：UCT
- 叶子评估：完整随机 rollout 直至终局
- 无历史偏差
- 无神经网络

### 3.3 `HeuristicMCTSAgent`
已存在。

定义：
- 树策略：UCT，支持现有的历史动作统计
- 未探索动作的选择可能使用基于历史值的 Gibbs/Softmax
- Rollout 策略可能使用历史引导的动作采样
- 同时更新树统计信息和全局历史统计信息

重要说明：
- 此类必须与 `PureMCTAgent` 清晰区分

### 3.4 `ValueGreedyAgent`
新类。

定义：
- 无树搜索
- 枚举当前状态的合法动作
- 对每个动作，计算下一个状态
- 编码下一个状态
- 使用价值网络为下一个状态评分
- 选择对当前角色预测价值最高的动作

目的：
- 孤立评估所学价值网络本身的质量
- 作为纯搜索与搜索+网络之间的有用基线

### 3.5 `NeuralValueMCTSAgent`
新的主要方法。

定义：
- 树策略：UCT
- 叶子评估：使用神经价值估计代替 rollout
- 终局节点仍使用精确的终局效用值
- 反向传播：对于双人零和游戏，使用 negamax 符号翻转

这是主要提出的方法。

### 3.6 可选：`HybridValueRolloutMCTSAgent`
可选但推荐。

定义：
- 在叶子节点直接使用价值网络
- 可选地在置信度低或深度低于阈值时执行短 rollout
- 通过配置标志控制

目的：
- 用于比较纯神经叶子评估与混合评估的有用消融实验

---

## 4. 必需的代码仓库结构调整

重构为更模块化的布局。

推荐的目标结构：

```text
src/
├── game_runner.py
├── gdl_parser.py
├── ggp_statemachine.py
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── random_agent.py
│   ├── pure_mct_agent.py
│   ├── heuristic_mcts_agent.py
│   ├── value_greedy_agent.py
│   ├── neural_value_mcts_agent.py
│   └── hybrid_value_rollout_mcts_agent.py
├── mcts/
│   ├── __init__.py
│   ├── tree_node.py
│   ├── selectors.py
│   ├── evaluators.py
│   ├── rollout.py
│   └── history_stats.py
├── encoding/
│   ├── __init__.py
│   ├── state_encoder.py
│   ├── fact_vector_encoder.py
│   ├── board_tensor_encoder.py
│   └── vocab.py
├── nn/
│   ├── __init__.py
│   ├── value_net.py
│   ├── dataset.py
│   ├── trainer.py
│   └── inference.py
├── experiments/
│   ├── __init__.py
│   ├── generate_dataset.py
│   ├── train_value_model.py
│   ├── run_match_set.py
│   ├── benchmark.py
│   ├── ablation.py
│   └── summarize_results.py
├── configs/
│   ├── agents/
│   ├── training/
│   └── experiments/
└── outputs/
```

如果立即进行完整重构风险太大，可以先将新模块实现在新目录中，并在 `ggp_agent.py` 中保留兼容性垫片。

---

## 5. 对现有文件的必要更改

## 5.1 `ggp_statemachine.py`
保持核心逻辑不变。添加编码器和新智能体所需的辅助方法。

### 必需的添加项

添加以下方法或等效方法：

```python
class GameStateMachine:
    def get_state_facts_as_strings(self, state) -> list[str]:
        """返回状态中所有事实的稳定字符串表示。"""

    def get_current_role(self, state):
        """如果可以推导出，返回回合制游戏中的当前角色，否则返回 None。"""

    def extract_board_facts(self, state) -> list:
        """返回类似格子的事实，如 cell(x,y,piece) 或其他棋盘事实。"""

    def get_role_index(self, role) -> int:
        """返回角色在 self.roles 中的索引。"""
```

### 附加要求

- 保留现有的 `legal` 和 `next` 缓存
- 如果尚未清晰公开，通过类似 `get_perf_stats()` 的方法公开缓存统计信息
- 确保状态字符串化是确定性的
- 如果状态事实存储在集合中，在导出到编码器逻辑之前对其进行排序

### 重要约束
不要破坏当前的搜索智能体。

---

## 5.2 `ggp_agent.py`
此文件当前包含多个智能体实现。

### 必需的操作
将智能体实现重构到 `agents/` 下的单独文件中，但临时保留导入兼容性。

### 兼容性要求
在 `ggp_agent.py` 中保留一个最小包装器，例如：

```python
from agents.random_agent import RandomAgent
from agents.pure_mct_agent import PureMCTAgent
from agents.heuristic_mcts_agent import HeuristicMCTSAgent
```

如果旧代码导入了 `MCTSAgent`，临时将其别名为 `HeuristicMCTSAgent`：

```python
MCTSAgent = HeuristicMCTSAgent
```

这可以避免破坏现有脚本。

---

## 5.3 `game_runner.py`
保留比赛执行逻辑，但为实验自动化进行扩展。

### 必需的添加项
支持：
- 使用可配置种子的重复比赛
- 将比赛结果保存到 JSON/CSV
- 可选的双方交换
- 实验元数据记录
- 智能体配置序列化

### 必需的新功能
实现或支持：

```python
run_match(agent_a, agent_b, game_file, seed, time_limit, ...)
run_match_series(agent_a_factory, agent_b_factory, n_games, swap_sides=True, ...)
```

运行器必须能从实验脚本中使用，而无需手动编辑。

---

## 6. 新的共享抽象

## 6.1 `BaseAgent`
为所有智能体创建一个清晰的基类。

```python
class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def meta_game(self, game, role, rules, startclock, playclock):
        pass

    def select_action(self, game, state, legal_actions, time_limit):
        raise NotImplementedError

    def cleanup(self):
        pass
```

所有智能体必须暴露一致的接口。

---

## 6.2 `TreeNode`
将 MCTS 节点逻辑移动到 `mcts/tree_node.py`。

### 必需的字段

```python
class TreeNode:
    state_key: str
    parent: TreeNode | None
    action_from_parent: object | None
    children: dict
    untried_actions: list
    visits: int
    total_value: float
    mean_value: float
    player_to_move: str | None
    is_terminal: bool
```

### 要求
- 通过键或引用存储状态，避免不必要地到处深度复制
- 支持延迟扩展
- 保持类的通用性，以便所有 MCTS 智能体可以重用

---

## 6.3 `LeafEvaluator`
创建一个叶子评估抽象。

```python
class LeafEvaluator:
    def evaluate(self, game, state, role, time_limit=None) -> float:
        raise NotImplementedError
```

实现以下具体评估器：

### `RandomRolloutEvaluator`
- 如果是终局：精确效用值
- 否则：随机 rollout 到终局

### `HeuristicRolloutEvaluator`
- 如果是终局：精确效用值
- 否则：历史引导的 rollout

### `ValueNetworkEvaluator`
- 如果是终局：精确效用值
- 否则：编码状态并运行价值模型

### `HybridEvaluator` (可选)
- 结合神经评估与短 rollout

此抽象是必需的，以便重用相同的 MCTS 核心。

---

## 7. 状态编码要求

实现两个编码器。

## 7.1 `FactVectorEncoder`
这是首先要构建的编码器，因为它是让完整流水线适用于任意 GDL 游戏的最简单方法。

### 目的
将符号状态转换为固定长度的向量。

### 必需的设计
- 从遇到的事实构建全局词汇表
- 每个事实字符串映射到一个整数索引
- 将状态编码为多热向量或计数向量
- 可选地附加元数据特征：
  - 当前角色的独热编码
  - 移动次数/回合数
  - 如果有用，可以包含终局标志

### 必需的类

```python
class FactVocabulary:
    def fit(states_or_fact_lists): ...
    def encode_fact(fact: str) -> int: ...
    def save(path): ...
    def load(path): ...

class FactVectorEncoder:
    def __init__(self, vocab: FactVocabulary, include_role=True, include_turn_features=True): ...
    def encode(self, state, game, role=None) -> np.ndarray: ...
```

### 要求
- 确定性排序
- 支持保存/加载
- 推理时缺失的事实需妥善处理
- 不依赖于特定游戏结构

### 推荐的首个实现
为简单起见使用密集 NumPy 数组。如果需要，以后可以添加稀疏表示。

---

## 7.2 `BoardTensorEncoder`
在 `FactVectorEncoder` 工作后实现。

### 目的
支持具有更结构化神经输入的棋盘类 GGP 游戏。

### 必需的行为
- 检测类似格子的事实，如 `cell(x,y,content)` 或可配置的模式
- 尽可能推断棋盘尺寸
- 将棋子/内容符号映射到通道或嵌入
- 输出适合 MLP/CNN/Transformer 风格价值模型的张量

### 必需的类

```python
class BoardTensorEncoder:
    def __init__(self, schema=None, include_player_plane=True, include_turn_features=True): ...
    def fit(self, samples): ...
    def encode(self, state, game, role=None) -> np.ndarray: ...
```

### 要求
- 针对不同游戏的模式/可配置映射
- 如果自动推断失败，允许按游戏配置模式
- 健壮地回退到 `FactVectorEncoder`

### 重要说明
此编码器不是第一个端到端里程碑所必需的，但后续实验需要它。

---

## 8. 数据集生成流水线

创建 `experiments/generate_dataset.py` 和 `nn/dataset.py`。

## 8.1 数据源
使用现有搜索智能体通过自我对弈或搜索对弈生成数据集。

### 必需的受支持生成器
- `PureMCTAgent` 自我对弈
- `HeuristicMCTSAgent` 自我对弈

不要从神经自我对弈引导开始。

## 8.2 样本定义
每个样本应至少包含：

```json
{
  "game_name": "...",
  "state_facts": ["...", "..."],
  "acting_role": "white",
  "value_target": 1.0,
  "ply_index": 12,
  "terminal": false
}
```

可选地也存储：
- 编码后的输入向量/张量
- 比赛 ID
- 最终得分字典
- 合法动作数量

## 8.3 价值目标定义
对于双人零和实验，从**当前行动角色的视角**定义目标：

- 赢 -> `+1.0`
- 平 -> `0.0`
- 输 -> `-1.0`

如果游戏返回 `[0, 100]` 范围内的分数，则进行一致归一化。

### 必需的实用函数

```python
def outcome_to_value(score_for_role, score_for_opponent) -> float:
    ...
```

使用确定性转换并记录它。

## 8.4 采样策略
至少支持两种模式：

### `all_states`
存储每局游戏中每个非终局状态。

### `subsampled_states`
从每局游戏中存储状态的随机子集，以减少时间相关性。

推荐的默认值：
- 小规模运行时保留所有状态
- 对于较大数据集，以 0.3 到 0.5 的速率进行子采样

## 8.5 输出格式
同时支持：
- JSONL：用于透明度和调试
- NPZ/PT：用于更快的训练

推荐的初始实现：
- 原始 JSONL 导出
- 预处理脚本构建编码后的训练数据集

---

## 9. 价值网络要求

创建 `nn/value_net.py`。

## 9.1 第一个必需的模型：`MLPValueNet`
首先构建此模型。

### 必需的接口

```python
class MLPValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 128), dropout=0.1): ...
    def forward(self, x):
        # 返回形状为 [B, 1] 且在 [-1, 1] 范围内的张量
```

### 必需的架构

推荐的默认值：
- Linear(input_dim, 256)
- ReLU
- Dropout
- Linear(256, 128)
- ReLU
- Dropout
- Linear(128, 1)
- Tanh

### 要求
- PyTorch 实现
- 兼容 CPU
- 用于单状态评估的推理辅助函数

## 9.2 第二个模型：`BoardValueNet`
稍后实现。

可能的选择：
- 用于标记化棋盘输入的小型 Transformer/注意力模型

这不是第一个里程碑，但保留代码结构以便日后添加。

---

## 10. 训练流水线要求

创建 `nn/trainer.py` 和 `experiments/train_value_model.py`。

## 10.1 数据集类
实现一个 PyTorch 数据集包装器。

```python
class ValueDataset(torch.utils.data.Dataset):
    def __init__(self, samples, encoder, game_adapter=None): ...
    def __len__(self): ...
    def __getitem__(self, idx):
        return x, y
```

## 10.2 训练脚本要求
训练脚本必须支持：

- 数据集路径
- 编码器类型
- 模型类型
- 训练/验证/测试集划分
- 批次大小
- 学习率
- 随机种子
- 训练轮数
- 保存路径

### 必需的默认值
- 优化器：AdamW
- 学习率：`1e-3`
- 权重衰减：`1e-4`
- 批次大小：`128`
- 损失函数：MSE 或 Huber
- 梯度裁剪：`1.0`
- 基于验证损失的早停

## 10.3 需要记录的指标
至少包括：
- 训练损失
- 验证损失
- 测试损失
- 符号准确率（`sign(pred) == sign(target)`）
- 如果容易实现，可以记录预测直方图/汇总统计

## 10.4 保存的工件
训练必须保存：
- 模型权重
- 配置 JSON/YAML
- 编码器/词汇表文件
- 训练指标 JSON

---

## 11. `ValueGreedyAgent` 实现要求

创建 `agents/value_greedy_agent.py`。

### 必需的构造函数

```python
class ValueGreedyAgent(BaseAgent):
    def __init__(self, name, role, value_model, encoder, device="cpu"):
        ...
```

### 必需的 `select_action` 逻辑

1. 枚举合法动作
2. 对于每个合法动作：
   - 计算下一个状态
   - 从当前角色视角编码下一个状态
   - 运行价值网络
3. 选择预测价值最大的动作
4. 确定性地或使用种子 RNG 打破平局

### 要求
- 支持 CPU 推理
- 如果有帮助，可以缓存编码后的状态
- 如果推理失败，安全回退到随机动作
- 可选地为调试记录每个动作的分数

---

## 12. `NeuralValueMCTSAgent` 实现要求

创建 `agents/neural_value_mcts_agent.py`。

## 12.1 核心思想
此智能体应尽可能重用现有 UCT 框架中的内容。

### 必需的搜索行为
- 选择：UCT
- 扩展：与标准 MCTS 相同
- 叶子评估：
  - 终局 -> 精确值
  - 非终局 -> 神经价值估计
- 反向传播：
  - 在双人零和游戏中，通过交替玩家回合使用符号翻转传播价值

## 12.2 构造函数

```python
class NeuralValueMCTSAgent(BaseAgent):
    def __init__(
        self,
        name,
        role,
        value_model,
        encoder,
        iterations=200,
        exploration_c=1.4,
        device="cpu",
        evaluator_mode="value",
        seed=None,
    ):
        ...
```

## 12.3 必需的辅助方法

```python
def _state_key(self, state) -> str: ...
def _select(self, node): ...
def _expand(self, node, game): ...
def _evaluate_leaf(self, game, state) -> float: ...
def _backpropagate(self, path, value): ...
```

## 12.4 要求
- 纯 `value` 模式无随机 rollout
- 默认根据访问次数选择根动作
- 可选地暴露按均值价值选择的模式
- 在固定种子下尽可能实现确定性行为
- 在模拟过程中不得错误地修改共享游戏状态

## 12.5 重要的工程约束
代码必须小心处理基于 Prolog 的状态转换。

推荐方法：
- 尽可能使用不可变状态表示
- 避免对复杂状态对象进行过度深度复制
- 始终通过 `GameStateMachine` 方法计算下一个状态

---

## 13. 可选 `HybridValueRolloutMCTSAgent`

仅在 `NeuralValueMCTSAgent` 工作后创建。

### 建议的行为
- 在叶子节点，默认直接使用神经价值
- 可选地执行 `k` 步短 rollout，然后用网络评估
- 或组合两个分数：

```python
combined = alpha * value_net_score + (1 - alpha) * rollout_score
```

### 参数
- `rollout_depth`
- `alpha`
- `use_history_rollout`

这主要用于消融实验。

---

## 14. 共享实验基础设施

创建 `experiments/run_match_set.py`。

## 14.1 必需的职责

- 从配置实例化智能体
- 运行重复比赛
- 在适当情况下交换双方
- 保存原始结果
- 计算聚合指标

### 必需的输出指标
每个对阵至少需要：
- 胜率
- 平局率
- 平均得分
- 平均移动时间
- 平均游戏长度
- 可选的状态机性能统计

### 必需的持久化
保存到：
- JSON：用于富含元数据的结果
- CSV：便于生成表格

---

## 15. 必需的实验

以下所有实验必须通过脚本/配置得到支持。

## 15.1 实验 A：基线强度排序
比较：
- `RandomAgent`
- `PureMCTAgent`
- `HeuristicMCTSAgent`
- `ValueGreedyAgent`
- `NeuralValueMCTSAgent`

必需的对阵：
- Random vs Random
- PureMCT vs Random
- HeuristicMCTS vs PureMCT
- ValueGreedy vs Random
- NeuralValueMCTS vs PureMCT
- NeuralValueMCTS vs HeuristicMCTS

## 15.2 实验 B：时间预算敏感性
对于每个主要搜索智能体，在不同的决策时间限制下进行评估。

必需的设置：
- 0.1秒
- 0.5秒
- 1.0秒
- 2.0秒

智能体：
- PureMCT
- HeuristicMCTS
- NeuralValueMCTS

## 15.3 实验 C：搜索预算敏感性
在固定的迭代次数下进行评估。

必需的设置：
- 50
- 100
- 200
- 500

智能体：
- PureMCT
- HeuristicMCTS
- NeuralValueMCTS

## 15.4 实验 D：数据集大小敏感性
使用从以下来源生成的数据集训练价值模型：
- 200 局游戏
- 500 局游戏
- 1000 局游戏
- 3000 局游戏

评估：
- 验证损失
- ValueGreedy 胜率
- NeuralValueMCTS 胜率

## 15.5 实验 E：编码器消融
比较：
- FactVectorEncoder + MLPValueNet
- BoardTensorEncoder + 合适的棋盘模型

如果棋盘模型尚未准备好，仍需构建代码结构以便日后进行此实验。

## 15.6 实验 F：缓存/性能研究
比较状态机缓存：
- 启用
- 禁用

跟踪：
- 挂钟时间
- 合法动作缓存命中率
- 下一个状态缓存命中率
- 对游戏结果的影响（如果有）

---

## 16. 必需的配置系统

添加 YAML 或 JSON 格式的简单配置文件。

### 必需的配置组

#### 智能体配置
示例：

```yaml
name: neural_value_mcts
iterations: 200
exploration_c: 1.4
evaluator_mode: value
model_path: outputs/models/value_net.pt
encoder_path: outputs/models/fact_vocab.json
device: cpu
```

#### 训练配置
示例：

```yaml
dataset_path: outputs/datasets/train.jsonl
encoder: fact_vector
model: mlp
batch_size: 128
learning_rate: 0.001
epochs: 20
weight_decay: 0.0001
```

#### 实验配置
示例：

```yaml
game: games/breakthrough.kif
n_games: 50
swap_sides: true
playclock: 1.0
seed: 42
```

---

## 17. Codex 应遵循的里程碑计划

严格按照此顺序实现。

## 里程碑 1：重构并保留当前行为

任务：
1. 将智能体类拆分到 `agents/`
2. 从 `ggp_agent.py` 为旧导入创建别名
3. 确保 `RandomAgent` 和 `PureMCTAgent` 与之前完全一致地工作

验收标准：
- 现有比赛仍可运行
- 基线行为不变

## 里程碑 2：提取可重用的 MCTS 组件

任务：
1. 创建 `TreeNode`
2. 创建叶子评估器抽象
3. 将 rollout 逻辑移动到可重用函数中
4. 在可行的范围内，使 `PureMCTAgent` 和 `HeuristicMCTSAgent` 使用共享的 MCTS 组件

验收标准：
- 两个基线搜索智能体仍然工作
- 评估器可以干净地交换

## 里程碑 3：实现编码 + 数据集生成

任务：
1. 添加 `FactVocabulary`
2. 添加 `FactVectorEncoder`
3. 添加使用 PureMCT 和 HeuristicMCTS 自我对弈的数据集生成脚本
4. 保存 JSONL 数据集

验收标准：
- 可以端到端生成训练数据集
- 数据集样本是确定性的且有效

## 里程碑 4：实现价值训练

任务：
1. 添加 `MLPValueNet`
2. 添加 `ValueDataset`
3. 添加训练器
4. 保存模型 + 词汇表 + 指标

验收标准：
- 可以从命令行运行训练
- 模型检查点成功保存

## 里程碑 5：实现 `ValueGreedyAgent`

任务：
1. 加载训练好的模型
2. 加载编码器/词汇表
3. 评估合法后继状态
4. 选择最佳动作

验收标准：
- 智能体可以在常规游戏运行器中运行
- 如果启用调试，比赛日志显示预测值

## 里程碑 6：实现 `NeuralValueMCTSAgent`

任务：
1. 集成 `ValueNetworkEvaluator`
2. 添加神经叶子评估 MCTS
3. 支持固定迭代预算
4. 如果运行器逻辑中已存在，支持时间预算

验收标准：
- 智能体至少在一种游戏上稳定运行
- 可以与 PureMCT 和 HeuristicMCTS 进行比较

## 里程碑 7：实验和报告脚本

任务：
1. 添加比赛系列运行器
2. 添加基准测试聚合
3. 添加数据集大小实验
4. 添加时间预算实验
5. 添加搜索预算实验

验收标准：
- 结果保存到 JSON/CSV
- 一个命令可以重现完整的实验块

---

## 18. 对 Codex 的编码要求

这些要求是强制性的。

### 18.1 一般要求
- 编写干净的 Python 3 代码
- 在合理的地方使用类型提示
- 为公共类/函数添加文档字符串
- 在可行的情况下保持与当前项目的向后兼容性
- 避免不必要的繁重依赖

### 18.2 PyTorch
- 所有神经网络模块使用 PyTorch
- 必须完全支持 CPU
- GPU 支持可选，代码可假定存在 CUDA，若无CUDA退回CPU

### 18.3 随机性控制
- 每个脚本应接受一个种子
- 在适用的情况下，对 Python、NumPy 和 PyTorch 一致地使用种子 RNG

### 18.4 日志记录
- 添加轻量级日志记录或基于打印的进度报告
- 实验脚本必须保存机器可读的输出

### 18.5 故障处理
- 如果神经模型加载失败，引发明确的错误
- 如果编码器无法处理某个状态，提供可操作的信息并失败
- 如果某个游戏不支持棋盘编码，允许回退到事实编码

---

## 19. 建议的命令行入口点

Codex 应提供类似于以下的可运行脚本。

### 数据集生成

```bash
python experiments/generate_dataset.py \
  --game path/to/game.kif \
  --agent pure_mct \
  --n-games 500 \
  --playclock 1.0 \
  --output outputs/datasets/game_puremct.jsonl
```

### 训练

```bash
python experiments/train_value_model.py \
  --dataset outputs/datasets/game_puremct.jsonl \
  --encoder fact_vector \
  --model mlp \
  --epochs 20 \
  --output-dir outputs/models/game_value_mlp
```

### 比赛评估

```bash
python experiments/run_match_set.py \
  --game path/to/game.kif \
  --agent-a neural_value_mcts \
  --agent-b pure_mct \
  --model-path outputs/models/game_value_mlp/model.pt \
  --encoder-path outputs/models/game_value_mlp/vocab.json \
  --n-games 50 \
  --playclock 1.0 \
  --swap-sides true \
  --output outputs/results/neural_vs_puremct.json
```

### 基准测试批处理

```bash
python experiments/benchmark.py --config configs/experiments/baseline_suite.yaml
```

---

## 20. 首个目标可交付成果

第一个完全可接受的可交付成果是：

1. 现有基线仍可运行
2. 可以从 `PureMCTAgent` 自我对弈生成数据集
3. `FactVectorEncoder` 可以工作
4. `MLPValueNet` 可以被训练
5. `ValueGreedyAgent` 可以在现有的游戏运行器中运行
6. `NeuralValueMCTSAgent` 可以对抗 `PureMCTAgent`
7. 实验脚本可以将比赛摘要输出到 JSON/CSV

这是最小的完整垂直切片。

---

## 21. 核心实现后的锦上添花改进

仅在核心流水线稳定后实施。

- 特定于棋盘的编码器和棋盘价值模型
- 混合神经+rollout 评估器
- MCTS 中的批量神经推理
- MCTS 的置换表
- 如果以后扩展到广义 AlphaZero，可以添加动作先验
- 多游戏训练支持
- 更丰富的模型选择和检查点比较

---

## 22. 给 Codex 的最终指示

优先考虑**端到端的工作功能**，而非理想的抽象。

实现必须首先交付一个稳定的实验流水线，包含以下五个智能体：

- `RandomAgent`
- `PureMCTAgent`
- `HeuristicMCTSAgent`
- `ValueGreedyAgent`
- `NeuralValueMCTSAgent`

不要过早过度设计。

首选的实现顺序是：

1. 基线重构
2. 共享 MCTS 抽象
3. 事实向量数据集流水线
4. MLP 价值训练
5. 价值贪婪智能体
6. 神经价值 MCTS 智能体
7. 实验自动化

生成的代码应易于运行、易于进行实验比较，并与现有的 GDL/Prolog 状态机架构兼容。