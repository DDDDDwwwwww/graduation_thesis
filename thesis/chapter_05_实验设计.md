# 第五章 实验设计

## 5.1 实验目标与验证层次

围绕第三章提出的 SDRPV 方法，本文将实验设计组织为三个相互衔接的验证层次。第一层是 baseline benchmark，用于回答 SDRPV 在统一搜索预算下相对于传统搜索基线和神经价值基线是否具有实际对局优势；第二层是 ablation benchmark，用于分析 teacher 监督与 residual 学习两个关键组成部分对最终性能的贡献；第三层是离线指标评估，用于从监督拟合角度检验模型是否真正学到了相对于基线搜索值的有效修正。

## 5.2 实验环境与数据来源

本文的实验环境建立在统一的 GGP 状态机之上，所有方法共享相同的规则执行机制、状态转移接口与终局评分规则。在线实验采用的游戏包括 `breakthrough`、`connectFour` 和 `hex`，它们共同覆盖了不同的棋盘规模、行动空间与博弈结构，因此能够较为全面地考察本文方法在多种棋盘型 GGP 任务中的适用性。

在线实验统一采用 fixed-time（固定时间）设置。具体而言，每一步决策的搜索时间预算设为 `0.5 s`，同时设置搜索迭代上限为 `120` 次；所有结果均以 `seed=242` 的运行留档为统计依据。若第 $m$ 步的实际搜索时间和搜索迭代数分别记为 $t_m$ 与 $k_m$，则实验预算约束满足

$$
t_m \le 0.5,\qquad k_m \le 120.
$$

采用这一口径的原因在于，它更接近真实对局中“有限思考时间”的使用场景，同时也避免了不同随机种子和不同预算模式混合带来的解释复杂度。

训练数据由统一的数据构建流程生成。本文离线训练样本来自 `connectFour` 的自对弈轨迹，且由 `pure_mct` 与 `heuristic_mcts` 混合生成。数据生成阶段采用 `mixed_heuristic_pure` 策略，配置比例为 `heuristic_mcts:pure_mct = 0.8:0.2`；在本次 `seed=42` 的实际运行中，对应为 `159:41` 局（约 `79.5\%:20.5\%`）。采用该混合策略的目的，是在保留较强启发式轨迹的同时引入纯搜索轨迹，降低单一行为策略导致的分布偏置，使样本覆盖弱启发式到强启发式的更宽状态分布，从而提升价值模型对不同搜索风格对手的稳健性。随后，系统在上述自对弈过程中采集中间状态作为候选样本，并对同一状态分别执行两档搜索标注：一档为低成本预算，用于生成基线值 `b`；另一档为更高预算，用于生成 teacher 值 `q_t`。随后，系统依据终局得分生成归一化终局监督 `z`，并从当前状态提取阶段特征 `phi`（如回合进度、占位率和合法动作规模等），同时记录样本来源、游戏类型与步骤位置等 `metadata`。在此基础上，数据管线将 `b`、`q_t` 与 `z` 统一映射到 $[-1,1]$ 数值区间，并执行去重、异常值过滤与字段完整性检查，最终得到可用于训练的标准化样本

$$x=\bigl(s,b,q_t,z,\phi,\text{metadata}\bigr).$$
经过上述清洗与标准化处理后，最终形成 `4732` 条有效样本，并划分为训练集、验证集和测试集，规模分别为 `3785`、`473` 和 `474`。若样本总数记为 $N=4732$，则三类划分占比近似为

$$
\frac{3785}{4732}\approx 0.800,\qquad
\frac{473}{4732}\approx 0.100,\qquad
\frac{474}{4732}\approx 0.100.
$$

这种构建方式使训练目标从一开始就面向本文的最终应用场景，即在有限搜索预算下，通过学习修正量提升叶节点评估质量。

【表 5-1 预留：实验对象与数据来源说明表。建议列出三类游戏、fixed-time 搜索预算、随机种子、样本划分规模以及三类实验对应的数据来源与输出内容。】

```latex
\begin{table}[htbp]
\centering
\caption{实验对象与数据来源说明}
\label{tab:experiment_objects_and_data_sources}
\begin{tabular}{p{2.4cm}p{5.0cm}p{4.1cm}}
\toprule
项目 & 具体设置或数据来源 & 对应输出内容 \\
\midrule
在线实验游戏 & \texttt{breakthrough}、\texttt{connectFour}、\texttt{hex} & 跨游戏 fixed-time 对局结果 \\
搜索预算 & 每步 $0.5\,\mathrm{s}$，迭代上限 $120$ 次 & baseline benchmark 与 ablation benchmark 的统一评测口径 \\
随机种子 & 数据构建使用 \texttt{seed=42}，在线留档使用 \texttt{seed=242} & 可复核的数据生成过程与结果统计配置 \\
训练样本来源 & \texttt{connectFour} 自对弈轨迹，\texttt{mixed\_heuristic\_pure}，\texttt{heuristic\_mcts:pure\_mct=159:41} & 标准化 SDRPV 训练样本 \\
样本划分规模 & 总计 $4732$ 条，训练/验证/测试分别为 $3785/473/474$ & 残差训练、模型选择与离线指标评估 \\
baseline benchmark & SDRPV 对 \texttt{pure\_mct}、\texttt{heuristic\_mcts}、\texttt{baseline\_token\_transformer} 的在线对局 & 分游戏胜率与三游戏总体胜率 \\
ablation benchmark & 完整方案、去残差方案、去 teacher 方案的在线对局 & 组件贡献分析与消融对比结果 \\
离线指标评估 & 测试划分上的模型预测、基线值与 teacher 值 & MAE、相关系数与误差改善量 \\
\bottomrule
\end{tabular}
\end{table}
```

## 5.3 模型角色与对比基线设置

本文在实验中考察的方法可分为两组。第一组是本文的主方法，即基于 SDRPV 的神经价值增强搜索。该方法采用棋盘 token 表示和 Transformer 价值网络，以残差学习方式估计相对于低成本搜索值的修正量，并在在线搜索中将该估计作为叶节点评估的补充信息。

第二组是对比基线，包括 `pure_mct`、`heuristic_mcts` 和 `baseline_token_transformer`。其中，`pure_mct` 代表不使用启发式信息和神经评估的纯 UCT 搜索；`heuristic_mcts` 在统一 MCTS 内核上引入历史统计驱动的动作采样与 rollout 引导，可视为更强的启发式搜索基线 \cite{finnsson2008simulation}；`baseline_token_transformer` 则代表与本文主方法具有相同表示范式、但不采用 SDRPV 残差学习目标的神经价值基线，其设置有助于辨别本文收益究竟来自统一表示本身，还是来自 SDRPV 的学习目标与搜索接入方式。

这样的基线组合具有明确分工：`pure_mct` 用于考察本文方法相对于最基础搜索的提升幅度，`heuristic_mcts` 用于检验本文方法对较强启发式搜索是否仍具有竞争力，`baseline_token_transformer` 则用于评估 SDRPV 相对于已有神经价值方法的增益。由此，baseline benchmark 可以从搜索、启发式和神经价值三个角度对本文方法进行完整定位。

## 5.4 Baseline Benchmark 设计

baseline benchmark 的目的是在统一预算和统一环境下比较 SDRPV 与三类基线的真实对局表现。实验在 `breakthrough`、`connectFour` 和 `hex` 三个游戏上进行，所有方法均采用 fixed-time 口径，即每步 `0.5 s` 搜索时间预算与 `120` 次搜索迭代上限，并以 `seed=242` 作为留档结果的统计依据。为保证分游戏结果的稳定性，每个游戏下的每组对抗配置均运行 `50` 局对弈。若记胜局数、平局数和总对局数分别为 $N_{\mathrm{win}}$、$N_{\mathrm{draw}}$ 和 $N_{\mathrm{all}}$，则本文采用的胜率统计可写为

$$
\mathrm{WinRate}
=
\frac{N_{\mathrm{win}}+0.5N_{\mathrm{draw}}}{N_{\mathrm{all}}}.
$$

在该实验中，本文方法采用 SDRPV 驱动的神经价值增强搜索；对比方法依次为 `pure_mct`、`heuristic_mcts` 和 `baseline_token_transformer`。前三者共同构成从纯搜索到启发式搜索再到神经价值搜索的递进对照，因此该 benchmark 可以清楚回答两个问题：第一，SDRPV 是否优于传统搜索基线；第二，SDRPV 相对于已有神经价值方法是否具有额外收益。

除以 SDRPV 为核心的主 baseline benchmark 外，本文还新增了一个补充 baseline benchmark，即固定 `baseline_token_transformer` 为对局主体，仅与 `pure_mct` 和 `heuristic_mcts` 进行对弈。该补充实验与主 benchmark 共用完全一致的 fixed-time 设置、随机种子与对局轮数，因此能够在“相同表示范式、相同传统搜索对手、相同预算约束”的条件下，直接比较 SDRPV 与神经价值基线对传统搜索基线的实际压制能力。换言之，主 benchmark 用于证明 SDRPV 整体更强，而补充 benchmark 则进一步说明这种优势不仅来自 token Transformer 表示带来的普遍收益，也来自 SDRPV 的残差学习目标及其搜索接入方式。

【图 5-1 预留：baseline benchmark 组织流程图。建议展示“模型加载 -> 搜索代理配置 -> fixed-time 对局执行 -> 单局结果汇总 -> 分游戏统计与总体统计”的流程。】

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.90\linewidth]{figures/chapter5/baseline_benchmark_pipeline.pdf}
\caption{baseline benchmark 组织流程图。}
\label{fig:baseline_benchmark_pipeline}
\end{figure}
```

【表 5-2 预留：主 baseline benchmark 与补充 baseline benchmark 的关系表。建议列出两组实验的 agent A、对手集合、控制变量以及各自回答的研究问题。】

```latex
\begin{table}[htbp]
\centering
\caption{主 baseline benchmark 与补充 baseline benchmark 的设计关系}
\label{tab:baseline_and_supplement_design}
\begin{tabular}{p{2.8cm}p{3.2cm}p{4.0cm}p{4.2cm}}
\toprule
实验组 & agent A & 对手集合 & 主要回答的问题 \\
\midrule
主 baseline benchmark & SDRPV & pure\_mct, heuristic\_mcts, baseline\_token\_transformer & SDRPV 在统一预算下是否整体优于传统搜索基线与神经价值基线 \\
补充 baseline benchmark & baseline\_token\_transformer & pure\_mct, heuristic\_mcts & 在相同传统搜索对手下，SDRPV 的收益是否超过神经价值基线本身 \\
\bottomrule
\end{tabular}
\end{table}
```

【图 5-2 预留：主 baseline benchmark 与补充 baseline benchmark 的关系示意图。建议用双路径对照展示 SDRPV 与 `baseline_token_transformer` 分别对接 `pure_mct`、`heuristic_mcts`，并在中间标出 `SDRPV vs baseline_token_transformer` 的直接比较关系。】

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.88\linewidth]{figures/chapter5/baseline_supplement_relation.pdf}
\caption{主 baseline benchmark 与补充 baseline benchmark 的关系示意图。}
\label{fig:baseline_supplement_relation}
\end{figure}
```

## 5.5 Ablation Benchmark 设计



ablation benchmark 的目的是拆解 SDRPV 中两个关键设计的作用，即较强搜索监督信号与残差学习目标。为此，实验比较完整方案、去除残差项的方案以及去除 teacher 监督的方案，借此分析最终性能提升究竟来自哪一部分设计，以及两者之间是否存在协同效应。


该实验与 baseline benchmark 保持完全一致的环境控制方式，同样在 `breakthrough`、`connectFour` 和 `hex` 上运行，同样采用每步 `0.5 s` 的 fixed-time 搜索预算与 `120` 次迭代上限，并同样以 `seed=242` 的结果作为统计依据。每个游戏下的每组配置均运行 `50` 局，以保证消融比较具有足够的稳定性。

通过这一设置，ablation benchmark 可以回答如下问题：如果不学习残差修正量，而是直接依赖较强搜索值进行监督，在线表现会发生怎样的变化；如果保留残差框架但移除 teacher 监督信号，模型是否仍能形成有效的搜索增强。也就是说，消融实验是用来解释 SDRPV 的性能来源和结构合理性。


【表 5-3 预留：ablation 配置说明表。建议列出完整方案、去除残差方案、去除 teacher 方案三类配置在监督目标、在线接入方式和预期作用上的区别。】

```latex
\begin{table}[htbp]
\centering
\caption{ablation 配置说明}
\label{tab:ablation_configurations}
\begin{tabular}{p{3.2cm}p{3.5cm}p{3.6cm}p{3.2cm}}
\toprule
配置 & 监督目标 & 在线接入方式 & 预期作用 \\
\midrule
\texttt{full\_residual\_teacher} & 学习相对 \texttt{b} 的 teacher 残差修正量 & 以 residual value 形式接入 value-full 搜索 & 同时利用强搜索监督与残差学习，作为完整主线方案 \\
\texttt{ablation\_no\_residual\_teacher} & 直接拟合 teacher 搜索值，不学习残差项 & 保持神经价值接入，但移除 ``基线值 + 修正量'' 结构 & 检验 residual 设计对稳定性与在线收益的贡献 \\
\texttt{ablation\_no\_teacher\_residual} & 学习不含 teacher 蒸馏的残差目标 & 保留 residual 接入框架，但移除强搜索监督信号 & 检验 teacher 搜索监督对修正方向的作用 \\
\bottomrule
\end{tabular}
\end{table}
```

## 5.6 离线指标与控制变量说明

离线指标实验的作用，是从监督学习角度验证残差模型是否真正学到了比低成本基线搜索值更接近较强搜索值的估计。本文主要报告相关性、排序相关性、相对于较强搜索值的绝对误差，以及相对于基线搜索值的误差改进幅度等指标。若测试集上的预测值记为 $\hat{v}_i$，对应 teacher 值记为 $q_{t,i}$，基线搜索值记为 $b_i$，则核心指标可写为

$$
\mathrm{MAE}_{\hat{v}}
=
\frac{1}{N}\sum_{i=1}^{N}\lvert \hat{v}_i-q_{t,i}\rvert,
$$

$$
\mathrm{MAE}_{b}
=
\frac{1}{N}\sum_{i=1}^{N}\lvert b_i-q_{t,i}\rvert,
$$

$$
\mathrm{Gain}_{\mathrm{MAE}}
=
\mathrm{MAE}_{b}-\mathrm{MAE}_{\hat{v}},
$$

以及

$$
\rho_{\mathrm{P}}
=
\mathrm{corr}(\hat{v},q_t),\qquad
\rho_{\mathrm{S}}
=
\mathrm{corr}\bigl(\mathrm{rank}(\hat{v}),\mathrm{rank}(q_t)\bigr).
$$

与仅报告训练损失相比，这组指标更能直接对应第三章提出的问题定义，即模型是否学会了有效修正量而不是简单拟合某个静态标签。

为了使 baseline benchmark、ablation benchmark 与离线指标分析之间具备可比性，本文对关键变量进行了统一控制。首先，所有在线实验共享同一状态机、同一 MCTS 主循环和相同的终局评分规则，不同方法之间仅改变叶节点评估策略。其次，所有在线实验统一采用 `breakthrough`、`connectFour`、`hex` 三个游戏，统一采用 fixed-time 口径，即每步 `0.5 s` 搜索预算与 `120` 次搜索迭代上限，统一采用 `seed=242` 和每组配置 `50` 局的设置。再次，神经价值基线与本文方法都在相同的搜索框架下接入 MCTS，从而保证比较对象是“价值评估质量及其搜索增益”，而不是外围实现差异。
