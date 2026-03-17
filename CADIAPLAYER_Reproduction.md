我用的是"D:\Anaconda\envs\MLcourse\python.exe"这个虚拟环境，修改代码时若 patch 失败，请先缩小上下文重试，不要整文件覆盖。
# CADIAPLAYER (AAAI 2008) 论文复现指南

[cite_start]这篇论文介绍了 CADIAPLAYER，它是 2007 年 AAAI 通用人工智能游戏（GGP）竞赛的冠军程序 [cite: 9, 25, 140][cite_start]。该系统的核心贡献在于证明了基于 UCT（Upper Confidence-bounds applied to Trees）的蒙特卡洛（MC）模拟在不需要任何先验领域知识的情况下，能替代传统带启发式评估函数的博弈树搜索 [cite: 7, 8, 23, 24]。

---

## 1. 系统架构与技术栈

你需要搭建与 GGP 服务器交互并能解析游戏规则的基础架构。原版系统包含以下三个主要组件：

* [cite_start]**HTTP 服务器**：作为一个外部常驻进程运行，负责监听传入端口并与 GGP 游戏服务器通信 [cite: 51, 52, 53][cite_start]。每当接收到新的游戏描述时，它会生成一个游戏引擎的实例 [cite: 54]。
* [cite_start]**游戏引擎（状态空间管理器）**：负责将 GDL（游戏描述语言）翻译成 Prolog 代码 [cite: 55][cite_start]。系统使用 **YAP Prolog** 将生成的代码编译成库，负责所有特定游戏的状态空间操作，包括生成和执行合法动作，以及检测和评估终局状态 [cite: 57, 58]。
* [cite_start]**AI 搜索模块**：用 C++ 编写，包含用于动作决策的搜索算法 [cite: 59, 60]。

---

## 2. 核心算法：UCT 蒙特卡洛树搜索



[cite_start]系统的核心动作选择机制是 UCT 算法 [cite: 71][cite_start]。你需要为游戏树中的每个状态-动作对记录平均回报值 $\mathcal{Q}(s,a)$ [cite: 73]。

### 2.1 针对不同玩家数量的策略调整
* [cite_start]**单人游戏（谜题）**：在准备时间（start-clock）内，优先使用内存增强型 IDA* (Memory Enhanced IDA*) 算法 [cite: 60, 61][cite_start]。如果找到部分解（得分大于 0），在游玩时间（play-clock）内继续使用该算法 [cite: 61][cite_start]。如果未找到解，则退回到使用 UCT 算法 [cite: 62]。
* [cite_start]**双人游戏**：Agent 可以配置为最大化双方分差，或者在最大化自身分数的同时，在平局时倾向于最小化对手的分数 [cite: 66]。
* [cite_start]**多人游戏**：为了简化运算，Agent 只关注最大化自身分数，忽略其他玩家的得分情况 [cite: 67, 68]。

### 2.2 UCT 算法核心细节
* [cite_start]**未采样动作优先**：如果当前状态的动作集合中存在从未被采样的动作（没有估计值），算法必须优先选择它们 [cite: 78]。
* **参数设置**：
    * [cite_start]**折扣因子 ($\gamma$)**：设定为 **0.99**，目的是让算法偏好更早获得的回报 [cite: 106]。
    * [cite_start]**探索参数 ($C$)**：在原版 CADIAPLAYER 竞赛版本中，用于调整 UCT 奖励权重的参数 $C$ 被设置为 **40** [cite: 81, 151]。
* **树的内存管理**：
    * [cite_start]每次执行一个非模拟的真实动作后，必须删除当前状态以上的搜索树部分，以防止内存耗尽 [cite: 91, 92]。
    * [cite_start]在每次模拟回合中，只将遇到的**第一个**新节点添加到内存中的搜索树里 [cite: 93, 108]。
* [cite_start]**对手建模**：必须为游戏中的每个对手建立独立的博弈树模型，来估计他们各自收到的回报 [cite: 101, 102][cite_start]。这些独立模型在模拟时控制对应玩家的 UCT 动作选择 [cite: 104]。

---

## 3. 关键创新：基于历史启发式的未探索动作选择

[cite_start]面对只有唯一正确应对方式的局面时，随机选择未探索动作会导致错误信息传回 UCT 树 [cite: 120, 122][cite_start]。作者引入了全局历史动作价值 $\mathcal{Q}_{h}(a)$ 来偏置未探索动作的选择 [cite: 131, 132]。

1. [cite_start]**记录全局收益**：除了 $\mathcal{Q}(s,a)$ 外，还要记录每个动作独立于状态的平均回报 $\mathcal{Q}_{h}(a)$ [cite: 131]。
2. [cite_start]**未探索动作的初始化**：如果一个动作从未被探索过，将其 $\mathcal{Q}_{h}(a)$ 初始值设定为 GGP 的最高分 **100**，以偏置早期的探索 [cite: 136]。
3. [cite_start]**吉布斯采样 (Gibbs Sampling)**：使用以下公式计算选择动作 $a$ 的概率 $\mathcal{P}(a)$ [cite: 134, 135]：
   $$\mathcal{P}(a)=\frac{e^{\mathcal{Q}_{h}(a)/\tau}}{\Sigma_{b=1}^{n}e^{\mathcal{Q}_{h}(b)/\tau}}$$
4. [cite_start]**温度参数 ($\tau$)**：在实验中，该参数被设置为 **10** [cite: 182]。

---

## 4. 算法 1 伪代码逻辑参考

[cite_start]参考论文中的 Algorithm 1，主流程伪代码如下 [cite: 85]：

```text
函数 search(qValues 引用数组):
1: 如果当前是终局状态 (isTerminal):
2:   遍历所有角色/玩家 i (getRoles):
3:     qValues[i] = 评估终局目标得分(goal(i))
4:   返回
5: 
6: 遍历所有角色/玩家 i:
7:   获取玩家 i 在当前状态的所有合法动作 (getMoves)
8:   根据玩家 i 的状态空间模型（StateSpaces[i]）选择一个动作 (selectMove)
9:   将选择的动作加入当前回合的动作集 (playMoves)
10:
11: 执行当前动作集，推进到下一个状态 (make(playMoves))
12: 递归调用 search(qValues)
13: 状态回退 (retract())
14: 
15: 遍历所有角色/玩家 i:
16:   应用折扣因子: qValues[i] = 0.99 * qValues[i]
17:   使用当前动作和 qValues[i] 更新玩家 i 的状态空间模型 (update)
18: 返回