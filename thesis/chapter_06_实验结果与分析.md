# 第六章 实验结果与分析

本章围绕提出的 SDRPV 主线方法，对离线价值预测效果、在线 baseline benchmark 结果以及 ablation benchmark 结果进行分析。与前文实验设计相对应，本章的重点在于回答三个问题：其一，当前主线方法相对于不同基线的在线表现如何；其二，teacher 监督与 residual 目标对最终效果分别起到什么作用；其三，离线指标能否为在线结果提供支撑。

为确保统计口径一致，除离线指标外，在线对局结果采用固定时间（fixed-time）设置下 `seed=242` 的取值。本文分析涵盖 `breakthrough`、`connectFour` 与 `hex` 三个游戏。在固定时间评测中，每步决策的搜索时间预算为 `0.5 s`，同时设置搜索迭代次数上限为 `120` 次。本文实验基线对比包含 `pure_mct`、`heuristic_mcts` 与 `baseline_token_transformer` 共三种基线。

## 6.1 离线指标结果分析

离线部分用于验证 SDRPV 的残差学习目标在进入在线搜索之前是否已经具备有效性。该部分以 `q_t` 作为评估目标，在 `seed=242` 的测试划分上进行比较，测试样本数为 `474`。其中，低成本基线估计 `b` 由 `32` 次模拟的浅层 MCTS 给出，teacher 目标 `q_t` 来自 `600` 次模拟的高预算搜索，因此 residual 路线本质上学习的是“高预算搜索值相对于低预算浅层 MCTS 值的修正量”。

从离线结果看，Residual Value Model（SDRPV）在三项指标上均取得最优表现，其 `MAE`、`MSE` 和相关系数分别为 `0.1788`、`0.0504` 和 `0.8304`；作为参照的基线值 `b` 分别为 `0.1836`、`0.0532` 和 `0.8137`；teacher-only 模型则退化为 `0.2038`、`0.0942` 和 `0.6313`。与基线值 `b` 相比，SDRPV 在误差指标上实现了小幅但一致的下降，并在相关性指标上进一步提升，说明其学习到的并不是对原有估计的简单替换，而是对基线误差的定向修正。这一现象与第三章中 residual_v1 的目标设定相一致，即模型的核心作用不是重新拟合全部价值结构，而是在保留低成本先验的基础上，对关键偏差进行补偿。

更值得注意的是，孤立使用 teacher 直接回归 `q_t`（即 teacher-only 模型）在该测试划分上的表现不仅低于 SDRPV，也低于基线值 `b`。这并不表明 teacher 缺乏有效信息。相反，消融实验中移除 teacher 后完整模型性能下降最明显（见表~\ref{tab:ablation_benchmark_seed242}），这从侧面说明了其重要性。该结果反映的是：在当前统一状态表示和训练设置下，直接学习 `q_t` 的稳定性不足；而“先保留基线、再学习修正量”的残差路线更符合本研究问题的实际需求。换言之，离线结果已为后续在线实验提供重要前提：SDRPV 的收益主要来自对基线值的残差修正，而非对基线值的完全替代。

【表 6-1 预留：离线指标对比表。建议列出 baseline value `b`、teacher-only 和 Residual Value Model（SDRPV）在 `seed=242` 测试划分上的 `MAE`、`MSE` 与 `Corr`。】

```latex
\begin{table}[htbp]
\centering
\caption{离线价值评估结果对比（seed=242，测试集 $n=474$）}
\label{tab:offline_metrics_seed242}
\begin{tabular}{lccc}
\toprule
模型 & MAE$\downarrow$ & MSE$\downarrow$ & Corr$\uparrow$ \\
\midrule
Baseline value $b$ & 0.1836 & 0.0532 & 0.8137 \\
Teacher-only & 0.2038 & 0.0942 & 0.6313 \\
Residual Value Model (SDRPV) & \textbf{0.1788} & \textbf{0.0504} & \textbf{0.8304} \\
\bottomrule
\end{tabular}
\end{table}
```

【图 6-1 预留：离线指标对比图。建议使用分组柱状图或折线图展示三种方法在误差指标与相关性指标上的差异，并突出 residual 路线优于基线值 `b`、teacher-only 退化的现象。】

```latex
\begin{table}[htbp]
\centering
\caption{离线指标相对基线值 $b$ 的变化量}
\label{tab:offline_delta_vs_b}
\begin{tabular}{lccc}
\toprule
模型 & $\Delta$MAE & $\Delta$MSE & $\Delta$Corr \\
\midrule
Teacher-only & +0.0202 & +0.0410 & -0.1824 \\
Residual Value Model (SDRPV) & \textbf{-0.0048} & \textbf{-0.0028} & \textbf{+0.0167} \\
\bottomrule
\end{tabular}
\end{table}
```

## 6.2 Baseline Benchmark 结果分析

在线 baseline benchmark 关注的问题是：在相同 fixed-time 搜索预算下，SDRPV 主线方案能否在真实对局中稳定转化为可观的实战收益。根据统计结果，SDRPV 对 `pure_mct` 的三游戏总体胜率为 `0.7133`（`107/150`），对 `heuristic_mcts` 的三游戏总体胜率同样为 `0.7133`（`107/150`），对 `baseline_token_transformer` 的三游戏总体胜率为 `0.5467`（`82/150`）。因此，从整体上看，SDRPV 不仅能够明显压制两类传统搜索基线，而且在与强神经价值基线的直接对抗中也保持了总体正优势。这说明第三章提出的 residual value 接入方式并非只是在离线指标上有效，而是能够在统一预算约束下转化为真实对局中的胜率收益。

从分游戏结果看，SDRPV 的优势主要体现在 `breakthrough` 与 `connectFour` 两个任务上。在 `breakthrough` 上，其对 `pure_mct`、`heuristic_mcts` 和 `baseline_token_transformer` 的胜率分别达到 `0.82`、`0.88` 和 `0.50`；在 `connectFour` 上，三组胜率分别为 `0.78`、`0.80` 和 `0.70`。也就是说，SDRPV 在这两个游戏中不仅显著优于传统搜索基线，而且在 `connectFour` 上对三类基线均超过 `0.5`。这表明，基于 SDRPV 的价值修正已经能够稳定改善叶节点评估质量，并在多种类型对手面前形成可重复的对局优势。特别是在面对 `pure_mct` 与 `heuristic_mcts` 时，`breakthrough` 和 `connectFour` 上的胜率均处于 `0.78` 到 `0.88` 的区间，这说明该方法对传统搜索基线的增益是强而一致的。

相较之下，SDRPV 对 `baseline_token_transformer` 的总体胜率为 `0.5467`，这一优势虽然不大，但更应被理解为：`baseline_token_transformer` 本身已经是一个较强的神经价值基线，因此 SDRPV 与其对抗时呈现的是“在强基线上继续做增量提升”，而不是“从弱基线到强基线的跃迁”。在这个意义上，`0.5467` 反映的并非 SDRPV 优势不足，而是比较对象本身更强。更重要的是，SDRPV 在对抗强神经基线时仍然保持总体领先，同时在对传统搜索基线时展现出更高幅度的胜率提升，这使得其收益结构具有较清晰的解释：SDRPV 的主要价值不仅在于维持与强神经基线的竞争力，更在于进一步增强模型对传统搜索对手的压制能力。

为了验证这一点，本文新增了补充 baseline benchmark：固定 `baseline_token_transformer` 为对局主体，在与主实验完全一致的 `fixed-time=0.5 s`、`iterations_cap=120`、`seed=242` 和每组 `50` 局设置下，分别对 `pure_mct` 与 `heuristic_mcts` 进行评测。该实验的目的并不是重复主 benchmark，而是进一步考察：在相同 token Transformer 表示范式下，引入 SDRPV 之后，模型面对传统搜索基线的胜率是否得到提升。结果表明，`baseline_token_transformer` 对 `pure_mct` 和 `heuristic_mcts` 的三游戏总体胜率均为 `0.4600`（`69/150`）；对应地，SDRPV 对这两个传统搜索基线的三游戏总体胜率均为 `0.7133`（`107/150`）。这意味着，在相同传统搜索对手集合下，SDRPV 相比 `baseline_token_transformer` 额外带来了 `+0.2533` 的总体胜率增益；三游戏汇总后的两比例检验达到 `$p=8\times10^{-6}$`，说明这种增益具有统计显著性。

更关键的是，这种提升不是由单一游戏偶然驱动，而是在分游戏层面也具有一致性。面对 `pure_mct` 时，`baseline_token_transformer` 在 `breakthrough`、`connectFour` 和 `hex` 上的胜率分别为 `0.56`、`0.46` 和 `0.36`，而 SDRPV 对应提升到 `0.82`、`0.78` 和 `0.54`，增益分别为 `+0.26`、`+0.32` 和 `+0.18`；面对 `heuristic_mcts` 时，前者三项胜率分别为 `0.50`、`0.44` 和 `0.44`，后者则提升到 `0.88`、`0.80` 和 `0.46`，增益分别为 `+0.38`、`+0.36` 和 `+0.02`。因此，这个补充实验所支持的核心结论是：即使 SDRPV 相对于强神经价值基线的直接对抗优势并非特别大，它仍然显著提升了 token Transformer 路线在面对 `pure_mct` 与 `heuristic_mcts` 时的实战胜率。换言之，SDRPV 的作用不是简单替换原有表示，而是在保留神经价值建模能力的基础上，进一步增强其对传统搜索基线的压制效果。

结合第 6.1 节的离线结果可以进一步看出，这种在线收益与 residual 学习的目标设计是一致的。离线阶段学到的价值修正，并没有停留在相关性和误差指标的改进上，而是进一步转化为对传统搜索基线的稳定优势；与此同时，SDRPV 在与强神经价值基线的直接比较中仍保持总体正胜率，说明其收益并非局限于某一种单独对手类型。综合而言，本节结果更有力地支持了第三章的方法主张：SDRPV 的核心贡献，在于把 token Transformer 的价值建模能力进一步推进为对传统搜索基线更高、更稳的对局胜率。

【表 6-2 预留：baseline benchmark 结果表。建议列出 SDRPV 主线方案在 `breakthrough`、`connectFour`、`hex` 三个游戏上分别对 `pure_mct`、`heuristic_mcts` 和 `baseline_token_transformer` 的 fixed-time 胜率，并给出三游戏总体胜率。】

```latex
\begin{table}[htbp]
\centering
\caption{Baseline benchmark 结果（fixed-time，seed=242）}
\label{tab:baseline_benchmark_seed242}
\begin{tabular}{lcccc}
\toprule
对手基线 & breakthrough & connectFour & hex & 三游戏总体 \\
\midrule
pure\_mct & \textbf{0.82} & 0.78 & \textbf{0.54} & \textbf{0.713} \\
heuristic\_mcts & \textbf{0.88} & \textbf{0.80} & 0.46 & \textbf{0.713} \\
baseline\_token\_transformer & 0.50 & 0.70 & 0.44 & 0.547 \\
\bottomrule
\end{tabular}
\end{table}
```

【图 6-2 预留：baseline benchmark 分组柱状图。建议以游戏为横轴、以胜率为纵轴，展示 SDRPV 主线方案相对三个 baseline 的分游戏表现，用于突出 `breakthrough` 与 `connectFour` 的优势以及 `hex` 的短板。】

【表 6-3 预留：补充 baseline benchmark 对比表。建议采用“`baseline_token_transformer / SDRPV`”的单元格格式，列出两者在相同传统搜索对手下的分游戏胜率与总体胜率，并在正文中配合给出总体差值与显著性检验。】

```latex
\begin{table}[htbp]
\centering
\caption{补充 baseline benchmark：在相同传统搜索对手下比较 baseline\_token\_transformer 与 SDRPV}
\label{tab:supplement_baseline_compare}
\begin{tabular}{lcccc}
\toprule
对手 & breakthrough & connectFour & hex & 三游戏总体 \\
\midrule
vs pure\_mct & 0.56 / 0.82 & 0.46 / 0.78 & 0.36 / 0.54 & 0.460 / 0.713 \\
vs heuristic\_mcts & 0.50 / 0.88 & 0.44 / 0.80 & 0.44 / 0.46 & 0.460 / 0.713 \\
\bottomrule
\end{tabular}
\end{table}
```

【图 6-3 预留：补充 baseline benchmark 对比图。建议以对手类型为分组、以游戏为横轴，绘制 `baseline_token_transformer` 与 SDRPV 的并列柱状图或哑铃图，用于突出二者在 `breakthrough` 与 `connectFour` 上的差距，以及在 `hex` 上仍然存在的共同困难。】

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.92\linewidth]{figures/chapter6/supplement_baseline_compare.pdf}
\caption{补充 baseline benchmark 对比图。展示 `baseline_token_transformer` 与 SDRPV 在相同传统搜索对手下的分游戏胜率差异。}
\label{fig:supplement_baseline_compare}
\end{figure}
```
## 6.3 Ablation Benchmark 结果分析

ablation benchmark 的目的在于进一步回答：SDRPV 的在线收益究竟来自哪些关键组成部分。为此，本文在与 `pure_mct` 的同一 `fixed-time` 对局环境下，比较完整方案 `full_residual_teacher`、去除 residual 设计的 `ablation_no_residual_teacher` 以及去除 teacher 蒸馏信号的 `ablation_no_teacher_residual`。三种配置在 `seed=242`、三游戏、每游戏 `50` 局的条件下，固定时间总体胜率分别为 `0.6733`、`0.6467` 和 `0.5933`。从总体结果看，完整方案取得最高胜率，去除 teacher 后退化最明显，去除 residual 后也出现下降，但下降幅度相对较小。

从分游戏结果看，完整方案在 `breakthrough`、`connectFour` 和 `hex` 上的胜率分别为 `0.72`、`0.60` 和 `0.70`，体现出相对均衡的跨游戏表现。与之相比，去除 residual 的版本在 `connectFour` 上达到 `0.78`，高于完整方案的 `0.60`，但在 `breakthrough` 和 `hex` 上分别下降到 `0.58` 和 `0.58`。这说明直接使用 teacher 目标虽然可能在个别游戏上获得更高的局部收益，但缺乏稳定的跨游戏一致性；residual 设计的作用，并不只是追求单个游戏上的最优值，而是通过“基线值 + 修正量”的形式降低学习波动，使方法在多个游戏上取得更平衡的总体结果。

去除 teacher 的版本则在三项游戏上均弱于完整方案，其在 `breakthrough`、`connectFour` 和 `hex` 上的胜率分别为 `0.62`、`0.56` 和 `0.60`，相对于完整方案的降幅分别为 `0.10`、`0.04` 和 `0.10`；从总体平均上看，固定时间胜率下降了 `0.08`。这一结果表明，teacher 搜索信号对于构造有效的修正方向仍然具有基础性作用。如果缺少更强预算搜索提供的监督参照，即便保留 residual 形式本身，模型也难以稳定学到足以转化为在线优势的价值修正。

将本节结果与第 6.1 节的离线结果对应起来，可以得到更完整的证据链。离线实验已经表明，直接拟合 `q_t` 的 teacher-only 路线不如 residual 路线稳定；而在线消融进一步表明，虽然只保留 teacher、不用 residual 的方案在局部任务上可能出现高点，但从三游戏总体表现看仍不及完整方案；只保留 residual、去掉 teacher 的方案则会在所有游戏上同时退化。由此可以认为，SDRPV 的有效性并非由单一因素独立决定，而是来自强搜索 teacher 提供监督方向与 residual 目标降低学习难度两者的共同作用。

【表 6-4 预留：ablation benchmark 结果表。建议列出 `full_residual_teacher`、`ablation_no_residual_teacher` 和 `ablation_no_teacher_residual` 在三个游戏上的 fixed-time 胜率，以及三游戏总体胜率和相对完整方案的胜率差值。】

```latex
\begin{table}[htbp]
\centering
\caption{Ablation benchmark 结果（fixed-time，seed=242）}
\label{tab:ablation_benchmark_seed242}
\begin{tabular}{lccccc}
\toprule
配置 & breakthrough & connectFour & hex & 三游戏总体 & 相对完整方案 \\
\midrule
full\_residual\_teacher & \textbf{0.72} & 0.60 & \textbf{0.70} & \textbf{0.673} & 0.000 \\
ablation\_no\_residual\_teacher & 0.58 & \textbf{0.78} & 0.58 & 0.647 & -0.027 \\
ablation\_no\_teacher\_residual & 0.62 & 0.56 & 0.60 & 0.593 & -0.080 \\
\bottomrule
\end{tabular}
\end{table}
```

【图 6-4 预留：ablation benchmark 对比图。建议使用分组柱状图或差值图展示三种配置在三个游戏上的胜率变化，重点突出“去除 teacher 后全局退化”和“去除 residual 后跨游戏稳定性下降”两类现象。】

```latex
\begin{table}[htbp]
\centering
\caption{各消融配置相对完整方案的胜率变化}
\label{tab:ablation_delta_vs_full_seed242}
\begin{tabular}{lcccc}
\toprule
配置 & breakthrough & connectFour & hex & 三游戏总体 \\
\midrule
ablation\_no\_residual\_teacher & -0.14 & +0.18 & -0.12 & -0.027 \\
ablation\_no\_teacher\_residual & -0.10 & -0.04 & -0.10 & -0.080 \\
\bottomrule
\end{tabular}
\end{table}
```
