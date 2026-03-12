# 毕业项目评估 Pipeline 设计指南

## 1. 设计目标
在**不改动现有底层代码**（`gdl_parser.py`、`ggp_statemachine.py`、`ggp_agent.py`、`game_runner.py`）的前提下，搭建一套可复用的评估 pipeline，用于：
- 评估当前已完成的 baseline（`RandomAgent`、`PureMCTAgent`、`MCTSAgent`）
- 无缝接入后续强化学习智能体
- 统一输出可用于论文的胜率、效率、稳定性与收敛结果

## 2. 当前代码基础与可直接复用的接口
你的现有实现已经具备评估所需的核心接口：
- `GameStateMachine(game_file)`：负责 GDL/Prolog 驱动的状态转移
- `Agent.select_move(game_machine, state, time_limit=None)`：统一决策接口
- `GameRunner(game_file, agents).run_match(...)`：可执行完整对局
- `GameStateMachine` 已支持：
  - `get_roles()`
  - `get_initial_state()`
  - `get_legal_moves()`
  - `get_next_state()`
  - `is_terminal()`
  - `get_goal()`
  - cache / perf 统计

这意味着后续所有评估都应采用“**外层新增实验脚本，内层复用现有类**”的方式完成，而不是重写底层逻辑。

## 3. 推荐的总体结构
新增一个独立评估层，建议结构如下：

```text
project/
├─ gdl_parser.py
├─ ggp_statemachine.py
├─ ggp_agent.py
├─ game_runner.py
├─ games/
├─ experiments/
│  ├─ configs/
│  │  ├─ baseline.yaml
│  │  ├─ rl_eval.yaml
│  ├─ run_match_batch.py
│  ├─ tournament.py
│  ├─ collect_metrics.py
│  ├─ aggregate_results.py
│  └─ plot_results.py
├─ logs/
│  ├─ raw/
│  ├─ summary/
│  └─ figures/
└─ models/
```

其中 `experiments/` 是新增评估层，负责组织比赛、记录结果、汇总指标、画图；底层 4 个核心文件不动。

## 4. Pipeline 的 5 个阶段

### 阶段 A：实验配置层
用配置文件统一定义：
- 游戏列表：`ticTacToe`、`connectFour`、`bonaparte`、以及你当前能稳定跑通的其他 GDL 游戏
- 对阵组合：
  - `Random vs Random`
  - `Random vs PureMCT`
  - `Random vs MCTS`
  - `PureMCT vs MCTS`
  - `RLAgent vs Random`
  - `RLAgent vs PureMCT`
  - `RLAgent vs MCTS`
- 每局预算：
  - `move_time_limit`
  - `iterations`
  - `rollout_depth_limit`
- 每组重复次数：如 30 / 50 / 100 局
- 随机种子列表
- 输出路径

这一层的作用是让实验**可复现**，并且方便后面做 ablation。

### 阶段 B：单局执行层
单局仍然调用现有 `GameRunner.run_match()`，但在外层加一个包装器，负责：
- 初始化 agent
- 设置时间预算 / 搜索参数
- 记录：
  - 游戏名
  - 对局编号
  - agent 名称与参数
  - 角色分配
  - 最终分数
  - winner / draw
  - 总步数
  - 总耗时
  - cache/perf 统计

这里建议不要依赖终端打印作为结果来源，而是把每局结果保存为一条结构化日志（JSONL 或 CSV）。

### 阶段 C：批量对局层
外层写一个 batch runner，对每个游戏、每种对阵组合、每个随机种子循环运行多局。

关键要求：
- **轮换角色位置**：例如双人博弈中，A/B 要交换先后手
- **固定随机种子**：保证不同算法对比公平
- **多次重复**：避免单局偶然性
- **失败局单独记录**：若某局因 Prolog 或规则异常中断，不直接删掉，而是记为 failed case

### 阶段 D：指标汇总层
从原始日志中汇总出论文核心指标：

#### 结果指标
- 胜率 / 平局率 / 失败率
- 平均终局得分
- 各游戏上的 macro 平均表现

#### 效率指标
- 平均每步决策时间
- 平均每局总耗时
- 平均 legal/next 调用次数
- cache hit rate

#### 稳定性指标
- 不同随机种子下胜率方差
- 不同对局之间得分方差
- 失败局比例

#### 训练型指标（给未来 RL）
- self-play loss 曲线
- policy loss / value loss
- episode reward
- 与固定 baseline 对战时的阶段性胜率

### 阶段 E：可视化与论文输出层
自动生成：
- 胜率对比柱状图
- 不同游戏上的平均得分表
- 决策时间箱线图
- cache hit rate 对比图
- RL 训练收敛曲线

最终汇总成：
- `summary.csv`
- `ablation.csv`
- `figures/*.png`

这些结果可以直接进入论文实验章节。

## 5. 你这个项目最适合的评估对象划分

### 第一类：正确性验证
目的：证明状态机和对局流程是可用的。

最小实验：
- `Random vs Random`
- `PureMCT vs Random`
- 在 Tic-Tac-Toe / 简单游戏上检查：
  - 游戏能否正常终止
  - score 是否合理
  - 非法动作是否未出现

这部分结果主要用于说明“系统能跑通”。

### 第二类：baseline 性能验证
目的：证明当前搜索型智能体优于随机策略。

建议主实验：
- `Random vs PureMCT`
- `Random vs MCTS`
- `PureMCT vs MCTS`

这里最重要的论文结论是：
- 搜索智能体显著优于随机基线
- 带历史启发的 `MCTSAgent` 相比 `PureMCTAgent` 是否更强、更稳、更省查询开销

### 第三类：后续 RL 智能体验证
目的：证明强化学习模块带来增益。

后续接入时，RL 智能体只需满足同一个接口：
- `select_move(game_machine, state, time_limit=None)`

这样它就能直接被现有 `GameRunner` 和新的 batch runner 调用。

主对比应设置为：
- `RLAgent vs Random`
- `RLAgent vs PureMCT`
- `RLAgent vs MCTS`
- `RL-guided MCTS vs PureMCT / MCTS`（若你后续做 AlphaZero 风格）

## 6. 推荐的实验协议

### 游戏选择
建议只选你现阶段能稳定运行的 3 类游戏：
- 简单：`ticTacToe`
- 中等：`connectFour`
- 一个分支因子更高或多玩家的游戏，如你现在测试过的 `bonaparte`

这样可以形成“低复杂度 - 中复杂度 - 高复杂度/多角色”的梯度。

### 对局数量
建议：
- 调试期：每组 10 局
- 正式实验：每组 50-100 局
- 若计算太慢：至少保证每组 30 局，并报告局数限制

### 公平性约束
必须控制：
- 相同游戏规则文件
- 相同 `move_time_limit`
- 相同种子集合
- 双人游戏交换先后手
- 多人游戏轮换角色分配

## 7. 关键日志字段设计
每局至少记录以下字段：

```text
match_id
game_name
seed
agent_role_mapping
agent_class_mapping
agent_param_mapping
move_time_limit
final_scores
winner
is_draw
num_steps
wall_clock_sec
legal_calls
legal_cache_hits
next_calls
next_cache_hits
failed
error_msg
```

如果是 RL 训练，还应额外记录：

```text
episode_id
training_step
policy_loss
value_loss
total_loss
self_play_return
model_checkpoint
```

## 8. 论文里最值得做的 ablation
由于你当前 `MCTSAgent` 已经包含几个明确可控的超参数，最适合做消融：
- `iterations`
- `exploration_constant`
- `discount_factor`
- `temperature`
- `rollout_depth_limit`
- `fallback_legal_threshold`
- 是否使用历史启发（`MCTSAgent` vs `PureMCTAgent`）

最推荐的 ablation 主题：
1. **搜索次数增加是否提升胜率**
2. **历史启发是否优于纯随机 rollout**
3. **高分支游戏中 fallback 机制是否提升效率**
4. **时间预算变化对性能的影响**

## 9. 与后续 RL 模块的衔接方式
未来新增 RL 智能体时，不要修改 `GameRunner`，而是新增一个 agent 类，例如：
- `PolicyAgent`
- `ValueGuidedMCTSAgent`
- `AlphaZeroStyleAgent`

只要它继续实现：
- `select_move(game_machine, state, time_limit=None)`

整个评估 pipeline 无需重构，只需要在配置文件中把新 agent 注册进去即可。这一点非常适合毕业项目，因为它能体现你的系统具有**模块化与可扩展性**。

## 10. 最终建议的落地顺序
1. 先做 `run_match_batch.py`，把 baseline 多局对战跑通
2. 再做 `collect_metrics.py`，把单局日志规范化保存
3. 再做 `aggregate_results.py`，输出胜率、得分、耗时、cache 指标
4. 最后做 `plot_results.py`，产出论文图表
5. RL 完成后，只新增 agent 与训练日志，不改评估主框架

## 11. 一句话总结
你的项目最合适的评估 pipeline，不是重写底层状态机或 runner，而是**在现有统一接口外层增加“配置 - 批量运行 - 结构化记录 - 指标汇总 - 可视化”五层实验框架**。这样既完全贴合你当前代码，也能自然承接后续强化学习智能体，并直接服务于论文实验章节。
