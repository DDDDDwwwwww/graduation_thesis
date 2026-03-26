下面是我结合你原始 `Fast_slow_valuenet.md` 的可落地框架，以及你调研报告里关于**不确定性门控、单次前向不确定度估计、系统工程
# 方案 B：基于不确定性门控的两阶段价值网络（Fast–Slow Value Net）

## 1. 目标

在现有 `neural_mcts:token_transformer` 基线之上，设计一个**两阶段价值评估框架**，解决当前实验中已经暴露出的核心矛盾：

* `TransformerValueNet` 具有较强表达能力，尤其在复杂游戏中更有潜力；
* 但其推理开销较大，会明显压缩固定时间预算下的 MCTS 搜索次数；
* 因而在简单游戏、低预算、或搜索尚浅时，重型价值网络未必总能带来稳定收益；
* 当前实验现象已经说明：随着时间预算增大，Transformer 模型的优势会逐渐显现；但在较小预算下，高成本推理本身就是瓶颈。

本方案的核心思想是：

> 不再对所有待评估节点都直接调用较慢的 Transformer 价值网络，而是先使用一个**快速粗评网络（Fast Value Net）**进行低成本评估；仅当节点满足“高不确定性 / 高重要性 / 预算允许”等条件时，才进一步调用**精评网络（Slow Value Net）**进行高质量评估。

这样做的目标不是替代 Transformer，而是让 Transformer 只在**真正值得花成本的节点**上工作，从而在固定预算下改善整体搜索效率与最终决策质量。该思路本质上属于一种**多保真度评估（multi-fidelity evaluation）**与**自适应计算路由（adaptive computational routing）**的结合。

---

## 2. 方法概述

整个两阶段价值评估器由三个部分组成：

1. **共享输入编码器**：`BoardTokenEncoder`
2. **快速粗评网络**：`FastValueNet`
3. **精评慢网络**：`SlowValueNet`

推荐保持整体结构与当前项目兼容，即：

* `FastValueNet`：`BoardTokenEncoder + MLPValueNet`
* `SlowValueNet`：`BoardTokenEncoder + TransformerValueNet`

其中，`FastValueNet` 负责：

* 快速输出粗略价值估计；
* 输出一个轻量级不确定度指标；
* 为门控规则提供依据。

`SlowValueNet` 负责：

* 对被筛选出的关键节点进行高保真价值评估；
* 作为最终高质量评估器参与 MCTS 反向传播。

---

## 3. 核心设计思想

### 3.1 为什么需要两阶段而不是单一 Transformer

当前基线已经表明，Transformer 在复杂局面上的表达能力是有效的，但其高推理成本会挤占搜索预算。MCTS 的效果依赖于“**评估精度**”与“**可展开节点数**”的平衡：

* 单纯使用 Slow Net：单次评估准确，但搜索树变浅；
* 单纯使用 Fast Net：搜索树更深，但评估偏差更大；
* 两阶段方案：在大多数普通节点上节省成本，在少数关键节点上保留高精评估。

因此，本方案追求的不是“每个节点都最准确”，而是：

> 在给定 wall-clock time 下，让整棵搜索树的资源分配更合理。

这也是本方案相较于原始单网络 neural MCTS 的主要创新点。

---

## 4. 模型结构

## 4.1 输入编码器：`BoardTokenEncoder`

沿用当前项目已有的 `BoardTokenEncoder`，保持与 `token_transformer` 基线一致。

要求：

* 对棋盘类状态输出 token 化表示；
* 保持位置编码机制；
* 能同时服务于 MLP 和 Transformer 两个价值网络；
* 尽量避免为了 Fast / Slow Net 分别定义两套不兼容输入。
* 复用 Preliminary_experimental\outputs\experiments\D_dataset_size_sensitivity 中的 2000 训练集版本的 token_transformer作为主版本的 slow_value_net

建议：

* 编码逻辑完全共享；
* 如果工程上可行，可缓存一次编码结果，避免同一叶节点先后调用 Fast/Slow 时重复编码。

---

## 4.2 快速粗评网络：`FastValueNet`

### 推荐实现

* 输入：`BoardTokenEncoder` 输出
* 结构：轻量 `MLPValueNet`
* 输出：

  * 价值预测 `v_fast ∈ [0,1]`
  * 不确定度指标 `u_fast ≥ 0`

### 设计要求

* 推理显著快于 `TransformerValueNet`
* 参数规模尽量小，优先保证吞吐量
* 能对“简单局面”和“明显优劣局面”做出低成本判断
* 能提供一个**单次前向传播即可得到**的不确定度指标
* 复用 Preliminary_experimental\outputs\experiments\D_dataset_size_sensitivity 中的 1000 训练集版本的 token_mlp作为主版本的 fast_value_net

### 推荐定位

`FastValueNet` 不是弱化版 Transformer，而是一个“低成本过滤器”：

* 它的任务不是在所有节点上做到最优；
* 它的任务是快速判断：**这个节点值不值得唤醒 Slow Net**。

---

## 4.3 精评慢网络：`SlowValueNet`

### 推荐实现

* 输入：`BoardTokenEncoder` 输出
* 结构：当前基线中的 `TransformerValueNet`
* 输出：价值预测 `v_slow ∈ [0,1]`

### 设计要求

* 训练方式尽量与现有 `token_transformer` 保持一致；
* 作为高质量但昂贵的评估器，仅在必要时调用；
* 初版中不额外增加复杂头部，优先保持基线稳定。

---

## 5. 不确定度设计（重点）

你调研报告里最重要的一个结论是：

> 不能使用会带来多次前向传播开销的传统不确定性估计方法（如 MC Dropout / Ensemble）作为 Fast Net 的主方案，否则 Fast Net 就不“fast”了。

因此，这里必须坚持一个原则：

> **FastValueNet 的不确定度估计必须是单次前向传播（single-pass）方法。**

### 5.1 可行方案分级
以实现 方案 B：双头方差网络 为主
#### 方案 A：最小可运行版本（不实现）

直接用
[
u_{\text{fast}} = 1 - 2|v_{\text{fast}} - 0.5|
]
作为不确定度近似。

解释：

* 当 `v_fast` 接近 0 或 1 时，说明局面更“确定”；
* 当 `v_fast` 接近 0.5 时，说明局面更难判断；
* 该方案**无需改动网络结构**，最容易落地。

优点：

* 实现最简单；
* 几乎无额外成本；
* 足够支撑第一轮实验与论文初稿。

缺点：

* 这只是“基于中性值距离的启发式不确定度”，不是真正校准过的 epistemic uncertainty。

---

#### 方案 B：双头方差网络（实现）

让 `FastValueNet` 输出两个头：

* 价值头：`v_fast`
* 方差头：`\sigma^2_fast`

训练时将价值预测视作高斯分布均值，使用负对数似然（NLL）损失：

[
\mathcal{L}_{\text{NLL}}
========================

\frac{(y-v_{\text{fast}})^2}{2\sigma_{\text{fast}}^2}
+
\frac{1}{2}\log \sigma_{\text{fast}}^2
]

其中：

* `y` 是目标 value；
* `\sigma^2_fast` 作为网络自估计的不确定度。

优势：

* 仍然是单次前向；
* 数学上更规范；
* 比简单的 `|v-0.5|` 更适合写进论文方法部分。

这是**最推荐的正式版本**：实现难度适中，理论表达也够完整。

---

#### 方案 C：扩展版本——证据深度学习（EDL）（不实现）

进一步让网络输出用于构造分布参数的多维量，并从中解析出 epistemic uncertainty。

这个方向理论上更漂亮，但实现和调参都更复杂。建议定位为：

* 不作为第一阶段主实现；
* 可在论文“扩展讨论”或后续改进中提及；
* 若前两版效果很好，不必强行继续上复杂 UQ。

---

## 6. 门控规则（Gating Rule）

## 6.1 基本形式

对叶节点状态 `s`，Fast Net 输出：

* `v_fast(s)`
* `u_fast(s)`

门控器决定是否调用 Slow Net：

[
\text{gate}(s)=
\begin{cases}
1, & \text{if need slow evaluation} \
0, & \text{otherwise}
\end{cases}
]

最终价值为：

[
v(s)=
\begin{cases}
v_{\text{fast}}(s), & \text{if gate}(s)=0 \
v_{\text{slow}}(s), & \text{if gate}(s)=1
\end{cases}
]

初版不建议做 `v_fast` 与 `v_slow` 的加权融合，因为那会引入额外超参数，削弱实验解释性。

---

## 6.2 三类门控信号
实现**组合门控**版本
### A. 基于不确定度的门控

当 `u_fast(s)` 较高时调用 Slow Net。

若使用启发式不确定度，可写为：

[
|v_{\text{fast}}(s)-0.5| < \tau
]

若使用双头方差网络，则写为：

[
u_{\text{fast}}(s) > \tau
]

---

### B. 基于节点重要性的门控

仅凭不确定度还不够，因为搜索早期可能有大量节点都“不确定”，若全部送入 Slow Net，会使方法退化回原始 Transformer 基线。调研报告已经明确指出，**固定阈值 + 不考虑节点重要性**很容易失败。

因此建议加入节点重要性条件，例如：

* 节点访问次数 `N(s)`；
* 或节点是否位于当前 principal variation 附近；
* 初版中最简单可行的是访问次数阈值。

即：

[
N(s) \ge N_0
]

---

### C. 基于预算的门控

还应加入“每步慢网络预算”约束，防止某一步 Slow Net 调用过多。

定义：

* 每步最多允许 `B_slow` 次慢评估；
* 当该预算耗尽时，即使节点不确定，也只能使用 `v_fast`。

这能让门控机制真正具备“成本感知”能力，而不是只看局面难度。

---

## 6.3 推荐主版本：组合门控（实现版本）

建议论文主版本采用：

[
\text{Use SlowValueNet if }
u_{\text{fast}}(s) > \tau
;\land;
N(s) \ge N_0
;\land;
b_{\text{slow}} < B_{\text{slow}}
]

其中：

* `u_fast(s)`：Fast Net 不确定度
* `\tau`：不确定度阈值
* `N_0`：访问次数阈值
* `B_slow`：每步慢评估预算
* `b_slow`：当前步已使用慢评估次数

这版相较原始方案更完善，因为它同时考虑了：

1. 节点是否难判断；
2. 节点是否重要；
3. 当前预算是否允许继续调用重型模型。

这也是最符合你调研报告逻辑的一版。

---

## 7. 与 MCTS 的集成方式

## 7.1 集成位置

本方案只修改 MCTS 的**叶节点评估（leaf evaluation）**阶段：

* Selection：不变
* Expansion：不变
* Evaluation：改为 Fast–Slow 两阶段评估
* Backup：不变

这样做的好处：

* 与现有 `NeuralValueMCTSAgent` 兼容；
* 代码侵入小；
* 容易与 `neural_mcts:token_transformer` 做公平对比；
* 能最大程度复用你现有实验框架。

---

## 7.2 推荐接口设计

```python
class TwoStageValueEvaluator:
    def __init__(
        self,
        encoder,
        fast_value_net,
        slow_value_net,
        uncertainty_type="margin",   # "margin" or "variance_head"
        gate_type="combined",
        tau=0.15,
        visit_threshold=4,
        slow_budget_per_move=16,
    ):
        ...

    def evaluate(self, state, game, node_visit_count=0, slow_budget_used=0):
        """
        Returns:
        {
            "value": float,
            "used_slow_net": bool,
            "v_fast": float,
            "u_fast": float,
            "v_slow": Optional[float]
        }
        """
```

---

## 7.3 评估流程

对叶节点 `s`：

1. 编码 `s`
2. 用 `FastValueNet` 得到 `v_fast, u_fast`
3. 检查组合门控条件
4. 若不触发：

   * 返回 `v_fast`
5. 若触发：

   * 调用 `SlowValueNet`
   * 返回 `v_slow`
6. 记录统计量：

   * 是否触发 slow
   * slow 调用次数
   * fast / slow 推理耗时
   * 当前步慢评估占比

---

## 8. 训练方案

## 8.1 总体原则

不建议一开始就端到端联合训练整个 gated system。为了控制变量与实现风险，推荐采用**分阶段训练**。

---

## 8.2 Slow Net 的训练

直接复用当前 `TransformerValueNet` 的训练流程即可：

* 训练数据：当前状态–结果样本
* 目标：预测局面价值
* 作用：作为高质量教师模型与最终精评网络

---

## 8.3 Fast Net 的训练

Fast Net 建议分两层目标训练：

### 第一层：正常 value 监督

使用与 Slow Net 相同的数据集，预测最终 value target。

### 第二层：蒸馏目标（推荐加入）

为了减少 Fast Net 的系统性偏差，建议加入对 Slow Net 输出的蒸馏损失。调研报告明确指出，Fast Net 最大风险不是“稍微不准”，而是“**错误但自信**”，这会直接污染搜索树。

因此可使用：

[
\mathcal{L}_{\text{fast}}
=========================

\lambda_1 \mathcal{L}*{\text{target}}
+
\lambda_2 \mathcal{L}*{\text{distill}}
+
\lambda_3 \mathcal{L}_{\text{uncertainty}}
]

其中：

* `L_target`：对真实 value target 的监督损失
* `L_distill`：拟合 Slow Net 输出的蒸馏损失
* `L_uncertainty`：若使用方差头，则为 NLL 或校准项

推荐初版可简化为：

[
\mathcal{L}_{\text{fast}}
=========================

0.5,\mathcal{L}*{\text{target}}
+
0.5,\mathcal{L}*{\text{distill}}
]

---

## 8.4 训练顺序建议

### 第一阶段

* 先训练 `SlowValueNet`
* 固定 Slow Net

### 第二阶段

* 训练 `FastValueNet`
* 加入对 Slow Net 的蒸馏

### 第三阶段（暂不实现）

* 若效果稳定，再考虑轻量微调门控阈值和预算参数
* 不建议在论文主版本中再做复杂 end-to-end 联训

---

## 9. 工程实现注意事项

调研报告里最值得重视的一点，是它提醒了一个非常现实的问题：

> 如果门控后只剩很少一批节点再去跑 Transformer，可能会出现 GPU 批处理碎片化、异构推理调度低效、整体 wall-clock 反而不降的情况。

因此，这里建议把工程分成两个版本：

### 9.1 版本 1：论文主实现

采用最简单、最稳妥的同步实现：

* 先做 Fast
* 若门控触发，再做 Slow
* 不强行引入异步队列或多线程批处理

理由：

* 更适合毕业论文与首轮实验；
* 更容易保证正确性；
* 可以先验证“方法本身是否有效”。

### 9.2 版本 2：系统优化扩展 （暂不实现）

若主方案有效，再考虑：

* FastQueue / SlowQueue 双队列
* 批量异步推理
* virtual loss
* 慢网络批量积攒后再统一调用

这些内容可以作为论文中的“系统优化扩展”或 future work，而不必一开始就全部实现。

---

## 10. 实验设计

## 10.1 实验目标

本方案需要回答以下问题：

1. 两阶段方案是否优于原始 `neural_mcts:token_transformer`？
2. 它的收益是否主要出现在**低预算场景**？
3. 改进是否来自合理的门控，而不是偶然因素？
4. 它是否真的降低了 Slow Net 调用比例和平均决策耗时？
5. Fast Net 的不确定度设计是否有效？

---

## 10.2 对比对象

至少包括：

* `random`
* `pure_mct`
* `neural_mcts:token_mlp`
* `neural_mcts:token_transformer`
* `two_stage_neural_mcts`

其中核心比较对象是：

* `Slow-only`：`token_transformer`
* `Fast-only`：`token_mlp`
* `Fast–Slow`：你的新方法

---

## 10.3 主基准实验

### 目的

比较整体强度。

### 推荐设置

* 游戏：

  * `ticTacToe`
  * `connectFour`
  * `breakthrough`
* `playclock = 0.7`
* `iterations = 120`
* 每组对局数：尽量不少于 `20`

### 指标

* 胜率
* 平局率
* 平均每步决策时间
* 平均每步搜索节点数
* Slow Net 调用次数 / 占比

---

## 10.4 时间预算敏感性实验（核心）

### 目的

验证该方法是否特别适合小预算场景。

### 推荐设置

* 游戏：优先 `breakthrough`
* `playclock ∈ {0.2, 0.7, 1.5}`
* `iterations = 120`

### 核心比较

* `token_transformer`
* `two_stage_neural_mcts`

### 重点观察

* 在 `0.2s`、`0.7s` 下是否优于 baseline；
* 预算越紧，新方法收益是否越明显；
* 慢网络调用比例是否明显下降。

这组实验最能直接回应你的方法动机。 

---

## 10.5 搜索预算敏感性实验

### 目的

研究不同搜索深度下，两阶段机制的收益是否稳定。

### 推荐设置

* 游戏：`breakthrough`
* `iterations ∈ {50, 120, 300}`
* `playclock = 0.7`

### 关注点

* 小搜索预算下，减少慢评估是否更有帮助；
* 大搜索预算下，方法是否仍保持不劣；
* 是否出现“更多预算被还给了树扩展”的现象。

---

## 10.6 门控消融实验

### 对比版本

1. `Fast-only`
2. `Slow-only`
3. `Uncertainty Gate only`
4. `Visit Gate only`
5. `Uncertainty + Visit`
6. `Uncertainty + Visit + Budget`（主版本）

### 目的

证明真正有效的是“组合门控”，而不是某个偶然配置。

---

## 10.7 不确定度设计消融

建议比较：

1. `margin uncertainty`：`|v_fast - 0.5|`
2. `variance head`
3. 可选：无不确定度，只按访问次数门控

### 目的

验证更规范的不确定度设计是否确实优于纯启发式版本。

---

## 10.8 运行开销与机制分析

这是非常重要的一组辅助实验。

### 需要统计

* Fast Net 调用次数
* Slow Net 调用次数
* Slow 调用占比
* 平均每步 Fast 耗时
* 平均每步 Slow 耗时
* 平均每步总决策耗时
* 平均搜索节点数
* 单位时间展开节点数

### 需要证明

* 新方法确实减少了慢网络调用；
* 节省下来的时间确实转化成了更多搜索；
* 胜率提升不是“偶然”，而是资源分配机制真正起作用。

---

## 11. 成功判据

若满足以下至少两条，即可认为方案成功：

1. 在至少一个复杂游戏（优先 `breakthrough`）上显著优于 `token_transformer`
2. 在 `0.2s` 或 `0.7s` 小预算下明显优于 baseline
3. Slow Net 调用比例显著下降
4. 平均决策时间下降或更可控，且性能不降反升
5. 组合门控优于单一门控
6. `variance head` 版本优于简单 `|v-0.5|` 启发式版本

---

## 12. 推荐实现优先级

## 第一阶段：最小可运行版本

* `FastValueNet = token_mlp`
* `SlowValueNet = token_transformer`
* 不确定度：`|v_fast - 0.5|`
* 门控：`uncertainty + visit`
* 主测游戏：`breakthrough`

## 第二阶段：主论文版本

* Fast Net 改为双头输出：value + variance
* 加入蒸馏训练
* 门控升级为：`uncertainty + visit + budget`

## 第三阶段：增强版本（暂不实现）

* 加异步批处理 / 双队列
* 做更强的系统优化分析
* 可补充更复杂 UQ（如 EDL）

---

## 13. 推荐默认配置

```python
fast_net = BoardTokenEncoder + 2-layer MLP
slow_net = BoardTokenEncoder + TransformerValueNet

uncertainty_type = "variance_head"   # 初版可先用 "margin"
gate_type = "combined"

tau = 0.15
visit_threshold = 4
slow_budget_per_move = 16

main_game = "breakthrough"
main_setting = {
    "playclock": 0.7,
    "iterations": 120
}
```

若想先尽快跑通第一版，可使用：

```python
uncertainty_type = "margin"
tau = 0.15
visit_threshold = 4
slow_budget_per_move = 999999   # 第一版先不限制 budget 也可以
```
