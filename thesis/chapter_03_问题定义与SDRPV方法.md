# 第三章 问题定义与 SDRPV 方法

## 3.1 问题定义

本文关注的问题是：在通用博弈游戏（General Game Playing, GGP）框架下，如何为蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）构建一个可复用、可集成且具有跨游戏适应能力的神经价值评估模块。与只面向某一固定棋类设计规则和特征的系统不同，GGP 强调在统一规则描述和状态机接口上处理多种游戏，因此状态表示、价值监督和搜索集成方式都必须尽可能保持通用 \cite{genesereth2005general,browne2012survey}。基于这一前提，本文并不试图直接从原始状态端到端学习一个完全替代搜索的价值函数，而是聚焦一个更符合研究目标的问题：如何在统一搜索框架之上构建一个能够稳定修正局面估计偏差、并最终接入 MCTS 的价值模型。

从系统定义来看，统一输入由状态事实集合、当前行动角色以及对局阶段信息构成，统一输出则是在给定搜索预算下对当前局面的标量价值估计。若直接以终局胜负或弱搜索回报作为唯一监督信号，模型往往会面临监督粗糙、噪声较大，以及与真实搜索调用场景不完全一致等问题。为此，本文将价值建模问题重新表述为如下形式：对于给定状态 $s$，先构造一个低成本基线估计 $b(s)$，再利用更强预算的搜索生成 teacher 信号 $q_t(s)$，最终训练神经网络学习对基线估计的修正量。这样，模型在推理阶段得到的估计值可以更接近强搜索结果，并进一步转化为在线搜索中的高质量叶节点评估。

用参数为 $\theta$ 的价值模型表示这一过程，则本文关注的核心预测形式可写为

$$
\hat{v}_{\theta}(s)=\mathrm{clip}\bigl(b(s)+f_{\theta}(s,\phi),-1,1\bigr),
$$

其中 $f_{\theta}(s,\phi)$ 表示模型学习到的修正量，$\phi$ 表示与局面阶段相关的辅助特征。相应地，训练目标可以概括为

$$
\theta^\ast=\arg\min_{\theta}\ \mathbb{E}_{s\sim \mathcal{D}}\left[\ell\bigl(\hat{v}_{\theta}(s),q_t(s)\bigr)\right],
$$

即在样本分布 $\mathcal{D}$ 上，使最终预测尽可能逼近强搜索产生的 teacher 评价。

这一问题定义体现出本文方法的两个核心出发点。其一，价值模型不是为了脱离搜索而单独存在，而是服务于搜索过程中的节点评估，因此监督信号应当尽量接近真实搜索需求。其二，本文的目标不是抽象地追求更复杂的网络结构，而是围绕更稳定地提高搜索质量来组织状态表示、监督构造和模型训练过程。基于此，本文提出 Search-Distilled Residual Progress-guided Value（SDRPV）方法，将强搜索信号、低成本基线估计和阶段信息统一纳入价值学习过程。

【图 3-1 预留：SDRPV 方法总体流程图。建议展示状态采样、SDRPV 样本构造、teacher-only 训练、residual_v1 训练以及在线搜索接入五个环节之间的关系。】

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.92\linewidth]{figures/chapter3/sdrpv_overview.pdf}
\caption{SDRPV 方法总体流程图。}
\label{fig:sdrpv_overview}
\end{figure}
```

## 3.2 SDRPV 样本的构造方式

SDRPV 方法的数据基础，是围绕状态价值建模任务统一构造的监督样本。本文以对局过程中的中间状态作为基本学习单元，而不是只以整局输赢作为监督对象，并在此基础上组织出统一的 SDRPV 样本格式。每条样本包含 `s`、`b`、`q_t`、`z`、`phi` 和 `metadata` 六类信息。记单条样本为

$$
x=\bigl(s,b,q_t,z,\phi,\text{metadata}\bigr).
$$

其中，`s` 表示状态主体，内部保存 `state_facts`、`acting_role`、`ply_index` 和 `terminal`；`b` 表示低成本 baseline value；`q_t` 表示更高预算 teacher 搜索得到的监督值；`z` 表示归一化后的终局监督；`phi` 表示与阶段相关的全局特征；`metadata` 则用于记录样本来源及相关转换信息。

这种构造方式的核心在于，为同一状态同时保留“弱估计”和“强估计”，从而为后续残差学习建立明确参照。本文采用浅层搜索结果构造基线估计，而以更高模拟次数搜索得到的结果作为 teacher 信号。换言之，teacher 监督并非凭空设定，而是来自明显强于基线估计的搜索预算，因此可以被看作对更强搜索评价该局面的近似刻画。与此同时，所有监督值统一映射到 $[-1,1]$ 区间，以保证 `b`、`q_t` 与 `z` 在同一数值尺度上进行训练、比较和误差分析。若原始目标值 $g(s)$ 取值于 $[0,100]$，则归一化形式写为

$$
\mathrm{norm}\bigl(g(s)\bigr)=2\cdot \frac{g(s)}{100}-1.
$$

因此，终局监督可表示为

$$
z(s)=2\cdot \frac{\mathrm{goal}(s)}{100}-1.
$$

除了价值标签本身，SDRPV 样本还显式编码了阶段信息，以帮助模型理解局面所处的搜索语境。本文没有将阶段信息停留在抽象描述层，而是将其表示为 `phi`。其内容包括 `move_progress`、`board_occupancy_ratio`、`piece_count_diff`、`legal_move_count` 和 `terminal_proximity_proxy` 等统计量。可将其统一写为

$$
\phi(s)=\bigl[\phi_1(s),\phi_2(s),\ldots,\phi_K(s)\bigr]^\top,
$$

其中每个分量都可以从当前状态直接计算得到，不依赖特定游戏的手工知识，因此既保持了方法的通用性，也为后续价值修正提供了低成本的全局上下文。

从方法设计角度看，SDRPV 样本并不是若干标签的简单拼接，而是围绕同一研究目标组织起来的统一监督结构：`s` 提供状态表示，`b` 给出低成本参照，`q_t` 提供强搜索监督，`z` 提供终局结果约束，`phi` 则提供阶段上下文。这样的样本结构为 teacher-only 学习和 residual 学习提供了共同基础，也使本文的方法从一开始就具有完整的闭环特征。

【表 3-2 预留：SDRPV 样本字段说明表。建议列出字段名、含义、来源、取值范围和在训练中的作用，例如 `b` 来自浅层搜索，`q_t` 来自 teacher 搜索，`phi` 为阶段特征。】

```latex
\begin{table}[htbp]
\centering
\caption{SDRPV 样本字段说明}
\label{tab:sdrpv_sample_fields}
\begin{tabular}{p{1.8cm}p{2.8cm}p{2.5cm}p{2.3cm}p{4.3cm}}
\toprule
字段 & 含义 & 来源 & 取值范围 & 在训练中的作用 \\
\midrule
\texttt{s} & 当前状态主体，包含事实集合、行动方、步数与终局标志 & 自对弈轨迹中的中间状态 & 结构化状态对象 & 作为统一状态编码输入，承载局面语义 \\
\texttt{b} & 低成本 baseline value & 浅层 MCTS 搜索 & $[-1,1]$ & 提供低成本参照，并用于构造残差目标 \\
\texttt{q\_t} & teacher 搜索监督值 & 高预算 MCTS 搜索 & $[-1,1]$ & 提供更强搜索监督，刻画目标局面价值 \\
\texttt{z} & 归一化终局监督 & 终局得分映射 & $[-1,1]$ & 提供终局结果约束，可作为辅助监督信号 \\
$\phi$ & 阶段特征向量 & 当前状态统计量 & $\mathbb{R}^{K}$ & 提供进度、占位率、合法动作数等全局上下文 \\
\texttt{metadata} & 样本来源与转换记录 & 数据构建管线 & 结构化键值字段 & 支持去重、分组分析与结果留档追溯 \\
\bottomrule
\end{tabular}
\end{table}
```

## 3.3 teacher-only 路线的作用

在 SDRPV 的整体训练链路中，teacher-only 路线承担的是验证 teacher 信号学习质量的任务。该阶段直接以 `q_t` 或 `z` 作为监督目标训练价值模型，其中主线设置优先使用 `q_t`，即优先让模型学习更强搜索如何评价中间状态，而不是只学习最终输赢。

这一阶段之所以重要，主要有两个原因。第一，它验证了当前状态编码方式和价值网络结构是否能够有效吸收搜索蒸馏信号。如果 teacher-only 路线本身无法收敛，那么后续 residual 设计也缺乏可靠基础。第二，它为 residual_v1 提供了一个自然对照。teacher-only 的目标是直接拟合强搜索价值，而 residual_v1 的目标是在基线估计已存在的前提下只学习修正量。通过比较两者，可以更清楚地判断模型性能提升究竟来自更强监督本身，还是来自残差式目标构造。

从方法组织角度看，teacher-only 训练采用统一的状态编码与价值回归框架，主线使用基于棋盘 token 的表示方式和 Transformer 价值模型，损失函数则采用对异常值更稳健的 Huber 损失。若 teacher-only 模型的输出记为 $\hat{q}_{\theta}(s)$，则其优化目标可以写为

$$
\mathcal{L}_{\mathrm{teacher}}
=
\frac{1}{N}\sum_{i=1}^{N}
\mathrm{Huber}\bigl(\hat{q}_{\theta}(s_i),q_t(s_i)\bigr).
$$

这样的设计使得搜索蒸馏监督、统一状态表示和后续残差学习可以构成一条连续的方法链，而不是彼此割裂的若干局部模块。teacher-only 路线是验证 SDRPV 方法论的必要中间环节。

## 3.4 residual_v1 的核心思想

如果说 teacher-only 解决的是能否学到 teacher 信号，那么 residual_v1 解决的则是在基线估计之上实现更稳定的超越。其核心目标定义为

\[
\Delta(s)=\mathrm{clip}(target(s)-b(s),-1,1),
\]

其中 `target` 在主线设置中默认为 `q_t`。模型在训练时并不直接输出最终价值，而是输出残差预测 \(\hat{\Delta}(s)\)；在推理阶段，再通过

\[
\hat{v}(s)=\mathrm{clip}(b(s)+\hat{\Delta}(s),-1,1)
\]

恢复最终价值估计。

这样的设计利用了基线估计已经具备的局面判断能力，使网络不必从零拟合整个价值函数，而是重点学习基线估计在何种状态上会系统性偏高或偏低。对于以提升搜索评估质量为目标的方法而言，这一目标比直接回归完整 value 更贴近真实需求。它并不要求模型在所有局面上重新建立完整价值排序，而是要求模型重点修正基线估计中最关键的偏差部分，从而以更低的学习难度获得更直接的性能收益。

进一步地，residual_v1 的训练损失可写为

$$
\mathcal{L}_{\mathrm{res}}
=
\frac{1}{N}\sum_{i=1}^{N}
\mathrm{Huber}\bigl(\hat{\Delta}_{\theta}(s_i),\Delta(s_i)\bigr).
$$

residual_v1 的另一个优势在于，它把基线估计是否真正修正直接纳入模型评估之中。除了残差训练本身，本文还通过相关系数、秩相关系数以及 `MAE(v_hat,target)` 相对 `MAE(b,target)` 的改善情况来检验最终预测值是否优于基线估计。对应的误差改善量可以写为

$$
\mathrm{Gain}_{\mathrm{MAE}}
=
\mathrm{MAE}\bigl(b,target\bigr)-\mathrm{MAE}\bigl(\hat{v},target\bigr),
$$

其中

$$
\mathrm{MAE}(u,target)=\frac{1}{N}\sum_{i=1}^{N}\lvert u(s_i)-target(s_i)\rvert.
$$

这样一来，residual_v1 不仅提出了一种新的训练目标，也为后续实验章节中的有效性分析提供了直接证据链。

【图 3-3 预留：residual_v1 训练目标示意图。建议展示 baseline 值 `b`、teacher 值 `q_t`、残差目标 `q_t-b` 与最终预测值 `b+\hat{\Delta}` 之间的关系。】

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.88\linewidth]{figures/chapter3/residual_v1_target.pdf}
\caption{residual\_v1 训练目标示意图。}
\label{fig:residual_v1_target}
\end{figure}
```

## 3.5 输入表示、模型形式与推理接口

为了让学习到的价值模型能够真正服务于 GGP 场景下的在线搜索，本节进一步说明模型在表示层、结构层和接口层上的具体设计。在输入表示方面，本文没有重新设计一套面向单一游戏的特征工程，而是采用 token 化状态编码思路，将棋盘内容、空间位置、当前玩家以及回合信息组织为结构化输入；在主线方案中，最终使用的是基于棋盘 token 的表示方式与 Transformer 价值模型的组合。这样的编码方式既能保留局面结构信息，又能够兼顾不同游戏之间的统一表示需求。

在模型形式上，本文并未追求额外复杂的专用结构，而是优先选择能够稳定承载统一状态表示、teacher 监督和残差目标的价值网络。这一选择与 SDRPV 的整体设计目标一致，即在保留 GGP 通用性的前提下，以尽可能清晰且可复现的方式实现稳定的价值修正能力。换言之，本文的方法创新不在于引入全新的网络骨干，而在于将状态表示、搜索蒸馏监督和残差学习目标组织成一条可复用的统一链路。

在推理接口方面，本文采用统一的价值预测接口。对于终局状态，系统直接依据游戏规则返回真实得分；对于非终局状态，则将编码结果送入价值网络得到标量估计。记编码器输出为 $x(s)$，则主线模型的推理形式可写为

$$
h(s)=\mathrm{Transformer}\bigl(x(s),\phi(s)\bigr),\qquad
\hat{v}(s)=\tanh\bigl(W h(s)+b\bigr).
$$

这种接口设计，既保证了预测逻辑与游戏规则保持一致，也使模型能够自然嵌入搜索框架，而不会破坏原有的终局判定机制。

## 3.6 与 MCTS 的集成方式

训练价值模型并不是本文的最终目的，将其稳定接入搜索流程并转化为可观测的搜索收益，才是 SDRPV 方法成立的关键。为此，本文设计了统一的神经价值搜索代理，并支持 `value` 与 `selective` 两种评估模式。前者对应 value-full 接入方式，即在叶节点评估时直接调用价值网络；后者对应选择性接入方式，即只在满足特定条件时使用神经网络，在不满足条件时退回到随机 rollout 或低成本评估器，并通过 `alpha`、`max_neural_evals_per_move` 和 `legal_move_threshold` 等参数控制调用强度。

对非终局叶节点 $s_\ell$，value-full 模式的评估可写为

$$
V_{\mathrm{leaf}}^{\mathrm{value}}(s_\ell)=\hat{v}(s_\ell).
$$

而 selective 模式可表示为

$$
V_{\mathrm{leaf}}^{\mathrm{sel}}(s_\ell)
=
I(s_\ell)\hat{v}(s_\ell)+\bigl(1-I(s_\ell)\bigr)V_{\mathrm{fallback}}(s_\ell),
$$

其中 $I(s_\ell)\in \{0,1\}$ 表示当前叶节点是否满足神经评估调用条件，$V_{\mathrm{fallback}}$ 表示回退评估器输出。

这种集成方式延续了深度网络与树搜索协同的基本思路，但并不照搬 AlphaZero 一类大规模自博弈闭环，而是更贴近本文的研究条件与方法目标 \cite{anthony2017thinking,silver2018general}。具体而言，本文保留统一的 GGP 状态机和既有 MCTS 主循环，只对叶节点评估器进行神经化扩展；同时将搜索蒸馏、基线修正与在线集成拆分为可独立验证的阶段，从而降低系统整体复杂度，并提升实验分析的可解释性。

从本文的方法组织来看，residual_v1 与 value-full 的组合构成了主线方案，而 selective 模式更多承担集成对照的作用。这一结构与前文分析相呼应：teacher-only 用于验证 teacher 信号可学习，residual_v1 用于提升相对基线估计的修正能力，而 value-full 则对应更直接、更稳定的在线部署方式。

【图 3-4 预留：在线搜索中的价值评估接入示意图。建议展示统一 MCTS 主循环、value-full 评估路径、selective 评估路径与 fallback 评估路径之间的关系。】

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.92\linewidth]{figures/chapter3/value_mcts_integration.pdf}
\caption{在线搜索中的价值评估接入示意图。}
\label{fig:value_mcts_integration}
\end{figure}
```

## 3.7 本章小结

本章围绕 SDRPV 方法的设计逻辑与实现路径，对问题定义、数据构造、训练目标和系统集成方式进行了系统说明。首先，本文将 GGP 场景下的价值建模问题重新表述为一个“以强搜索为 teacher、以低成本基线估计为参照、以残差修正为目标”的学习问题；其次，构建了包含 `s`、`b`、`q_t`、`z` 和 `phi` 的 SDRPV 样本，为后续训练提供统一监督基础。

在此基础上，本文进一步说明了 teacher-only 路线与 residual_v1 路线在整体方法中的不同职责：前者用于验证搜索蒸馏信号是否可学习，后者用于显式学习 `target-b` 的修正量，从而更直接地服务于价值修正这一研究目标。随后，本文介绍了输入表示、模型形式与统一推理接口，并说明了模型如何接入 MCTS 主流程，形成可用于在线 benchmark 的搜索增强组件。

总体而言，SDRPV 方法的优势主要体现在三个方面：其一，监督信号更贴近真实搜索需求；其二，残差目标更贴近价值修正这一核心任务；其三，整个流程能够在统一的 GGP 与 MCTS 框架内形成完整闭环。下一章将进一步介绍系统层面的模块组织与实现细节，第五章和第六章则分别对实验设计与方法效果进行验证。
