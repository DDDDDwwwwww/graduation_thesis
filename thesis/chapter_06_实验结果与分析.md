# 第六章 实验结果与分析

本章围绕提出的 SDRPV 主线方法，对离线价值预测效果、在线 baseline benchmark 结果以及 ablation benchmark 结果进行分析。与前文实验设计相对应，本章的重点在于回答三个问题：其一，当前主线方法相对于不同基线的在线表现如何；其二，teacher 监督与 residual 目标对最终效果分别起到什么作用；其三，离线指标能否为在线结果提供支撑。

为确保统计口径一致，除离线指标外，在线对局结果采用固定时间（fixed-time）设置下 `seed=242` 的取值。本文分析涵盖 `breakthrough`、`connectFour` 与 `hex` 三个游戏。在固定时间评测中，每步决策的搜索时间预算为 `0.5 s`，同时设置搜索迭代次数上限为 `120` 次。本文实验基线对比包含 `pure_mct`、`heuristic_mcts` 与 `baseline_token_transformer_2000` 共三种基线。

## 6.1 离线指标结果分析

离线部分用于验证 SDRPV 的残差学习目标在进入在线搜索之前是否已经具备有效性。该部分以 `q_t` 作为评估目标，在 `seed=242` 的测试划分上进行比较，测试样本数为 `474`。其中，低成本基线估计 `b` 由 `32` 次模拟的浅层 MCTS 给出，teacher 目标 `q_t` 来自 `600` 次模拟的高预算搜索，因此 residual 路线本质上学习的是“高预算搜索值相对于低预算浅层 MCTS 值的修正量”。

从离线结果看，Residual Value Model（SDRPV）在三项指标上均取得最优表现，其 `MAE`、`MSE` 和相关系数分别为 `0.1788`、`0.0504` 和 `0.8304`；作为参照的基线值 `b` 分别为 `0.1836`、`0.0532` 和 `0.8137`；teacher-only 模型则退化为 `0.2038`、`0.0942` 和 `0.6313`。与基线值 `b` 相比，SDRPV 在误差指标上实现了小幅但一致的下降，并在相关性指标上进一步提升，说明其学习到的并不是对原有估计的简单替换，而是对基线误差的定向修正。这一现象与第三章中 residual_v1 的目标设定相一致，即模型的核心作用不是重新拟合全部价值结构，而是在保留低成本先验的基础上，对关键偏差进行补偿。

更值得注意的是，孤立使用 teacher 直接回归 `q_t`（即 teacher-only 模型）在该测试划分上的表现不仅低于 SDRPV，也低于基线值 `b`。这并不表明 teacher 缺乏有效信息——消融实验中移除 teacher 导致完整模型性能下降最明显（见表 X），恰恰证明了其重要性。该结果反映的是：在当前统一状态表示和训练设置下，直接学习 `q_t` 的稳定性不足；而“先保留基线、再学习修正量”的残差路线更符合本研究问题的实际需求。换言之，离线结果已为后续在线实验提供重要前提：SDRPV 的收益主要来自对基线值的残差修正，而非对基线值的完全替代。

【表 6-1 预留：离线指标对比表。建议列出 baseline value `b`、teacher-only 和 Residual Value Model（SDRPV）在 `seed=242` 测试划分上的 `MAE`、`MSE` 与 `Corr`。】

【图 6-1 预留：离线指标对比图。建议使用分组柱状图或折线图展示三种方法在误差指标与相关性指标上的差异，并突出 residual 路线优于基线值 `b`、teacher-only 退化的现象。】

## 6.2 Baseline Benchmark 结果分析

在线 baseline benchmark 用于回答一个更直接的问题，即在相同 fixed-time 搜索预算下，SDRPV 主线方案能否在真实对局中超过传统搜索基线与神经价值基线。根据筛选后的统计结果，SDRPV 主线方案对 `pure_mct` 的三游戏总体胜率为 `0.7133`（`107/150`），对 `heuristic_mcts` 的三游戏总体胜率同样为 `0.7133`（`107/150`），对 `baseline_token_transformer_2000` 的三游戏总体胜率为 `0.5467`（`82/150`）。这表明在本章限定的统计口径下，SDRPV 主线方案对三类基线均保持了总体上的正优势，但不同游戏之间的收益分布并不均衡。

从分游戏结果看，`breakthrough` 与 `connectFour` 是 SDRPV 主线方案优势最明显的两个任务。在 `breakthrough` 上，其对 `pure_mct`、`heuristic_mcts` 和 `baseline_token_transformer_2000` 的胜率分别为 `0.82`、`0.88` 和 `0.50`；在 `connectFour` 上，三组胜率分别为 `0.78`、`0.80` 和 `0.70`。其中，`connectFour` 上对三类基线均超过 `0.5`，说明 residual value 接入 MCTS 后能够稳定提升叶节点评估质量；`breakthrough` 上则对两类传统搜索基线形成了显著优势，而对 token Transformer 基线取得了 0.50 的胜率——考虑到该基线本身是专门设计的强对比方法，这一结果意味着 SDRPV 在与强神经基线的直接对比中能够保持不落下风，同时在对传统基线的提升幅度上表现更为突出。

与前两项游戏相比，`hex` 明显构成了当前方法的薄弱环节。SDRPV 主线方案在 `hex` 上对 `pure_mct` 的胜率为 `0.54`，但对 `heuristic_mcts` 和 `baseline_token_transformer_2000` 的胜率分别下降到 `0.46` 和 `0.44`。若将三类 baseline 在同一游戏上的结果取平均，则 `breakthrough`、`connectFour` 和 `hex` 的平均胜率分别为 `0.7333`、`0.7600` 和 `0.4800`。这一结果说明，当前方法虽然已经在总体上达到“对三类 baseline 均超过 `0.5`”的要求，但跨游戏适应能力仍存在明显差异，其主要短板集中在 `hex`。因此，本章对 baseline benchmark 的结论不应被表述为“SDRPV 在所有游戏上都形成稳定优势”，而应更准确地概括为：SDRPV 已经在 `breakthrough` 与 `connectFour` 上展现出较强竞争力，并在三游戏总体统计上保持领先，但在 `hex` 上的泛化表现仍需进一步增强。

结合第 6.1 节的离线结果可以看出，residual 学习确实为在线搜索带来了可转化的收益，但这种收益并非对所有游戏结构均等释放。对于 `breakthrough` 和 `connectFour`，离线阶段学到的价值修正能够较好地转化为在线对局优势；而在 `hex` 上，当前修正能力尚不足以在更强的启发式或神经基线面前形成稳定领先。由此可见，第三章所提出的“以低成本 baseline 为参照、以 residual 为核心”的方法路线在整体上是成立的，但其跨游戏稳定性仍然是需要正视的边界条件。

【表 6-2 预留：baseline benchmark 结果表。建议列出 SDRPV 主线方案在 `breakthrough`、`connectFour`、`hex` 三个游戏上分别对 `pure_mct`、`heuristic_mcts` 和 `baseline_token_transformer_2000` 的 fixed-time 胜率，并给出三游戏总体胜率。】

【图 6-2 预留：baseline benchmark 分组柱状图。建议以游戏为横轴、以胜率为纵轴，展示 SDRPV 主线方案相对三个 baseline 的分游戏表现，用于突出 `breakthrough` 与 `connectFour` 的优势以及 `hex` 的短板。】

## 6.3 Ablation Benchmark 结果分析

ablation benchmark 的目的在于进一步回答：SDRPV 的在线收益究竟来自哪些关键组成部分。为此，本文在与 `pure_mct` 的同一 `fixed-time` 对局环境下，比较完整方案 `full_residual_teacher`、去除 residual 设计的 `ablation_no_residual_teacher` 以及去除 teacher 蒸馏信号的 `ablation_no_teacher_residual`。三种配置在 `seed=242`、三游戏、每游戏 `50` 局的条件下，固定时间总体胜率分别为 `0.6733`、`0.6467` 和 `0.5933`。从总体结果看，完整方案取得最高胜率，去除 teacher 后退化最明显，去除 residual 后也出现下降，但下降幅度相对较小。

从分游戏结果看，完整方案在 `breakthrough`、`connectFour` 和 `hex` 上的胜率分别为 `0.72`、`0.60` 和 `0.70`，体现出相对均衡的跨游戏表现。与之相比，去除 residual 的版本在 `connectFour` 上达到 `0.78`，高于完整方案的 `0.60`，但在 `breakthrough` 和 `hex` 上分别下降到 `0.58` 和 `0.58`。这说明直接使用 teacher 目标虽然可能在个别游戏上获得更高的局部收益，但缺乏稳定的跨游戏一致性；residual 设计的作用，并不只是追求单个游戏上的最优值，而是通过“基线值 + 修正量”的形式降低学习波动，使方法在多个游戏上取得更平衡的总体结果。

去除 teacher 的版本则在三项游戏上均弱于完整方案，其在 `breakthrough`、`connectFour` 和 `hex` 上的胜率分别为 `0.62`、`0.56` 和 `0.60`，相对于完整方案的降幅分别为 `0.10`、`0.04` 和 `0.10`；从总体平均上看，固定时间胜率下降了 `0.08`。这一结果表明，teacher 搜索信号对于构造有效的修正方向仍然具有基础性作用。如果缺少更强预算搜索提供的监督参照，即便保留 residual 形式本身，模型也难以稳定学到足以转化为在线优势的价值修正。

将本节结果与第 6.1 节的离线结果对应起来，可以得到更完整的证据链。离线实验已经表明，直接拟合 `q_t` 的 teacher-only 路线不如 residual 路线稳定；而在线消融进一步表明，虽然只保留 teacher、不用 residual在局部任务上可能出现高点，但从三游戏总体表现看仍不及完整方案；只保留 residual、去掉 teacher则会在所有游戏上同时退化。由此可以认为，SDRPV 的有效性并非由单一因素独立决定，而是来自强搜索 teacher 提供监督方向与residual 目标降低学习难度两者的共同作用。

【表 6-3 预留：ablation benchmark 结果表。建议列出 `full_residual_teacher`、`ablation_no_residual_teacher` 和 `ablation_no_teacher_residual` 在三个游戏上的 fixed-time 胜率，以及三游戏总体胜率和相对完整方案的胜率差值。】

【图 6-3 预留：ablation benchmark 对比图。建议使用分组柱状图或差值图展示三种配置在三个游戏上的胜率变化，重点突出“去除 teacher 后全局退化”和“去除 residual 后跨游戏稳定性下降”两类现象。】

## 6.4 本章小结

本章在统一统计口径下，对 SDRPV 的离线与在线效果进行了集中验证。离线结果表明，residual 路线在 `seed=242` 测试划分上同时优于基线值 `b` 与 teacher-only，说明“学习修正量”比“直接替代 baseline”更适合当前问题设定。在线 baseline benchmark 结果进一步表明，SDRPV 主线方案已经能够在 `breakthrough` 与 `connectFour` 上形成较强优势，并在三类 baseline 上保持总体胜率超过 `0.5`；但其在 `hex` 上的表现仍不稳定，说明跨游戏适应能力尚未完全成熟。ablation benchmark 则说明，teacher 搜索监督和 residual 目标二者缺一不可：前者决定修正方向，后者决定跨游戏稳定性。综合而言，第三章提出的 SDRPV 方法路线已经得到较充分支持，但其优势目前更集中体现在“总体有效、部分游戏显著受益”，而不是“所有游戏上均匀提升”。
