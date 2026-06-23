## 引言

自动化渗透测试的目标,是端到端地复现人类红队对目标的完整动作:枚举暴露面、定位可利用的弱点、获取立足点、提升权限,直至取得 root。大语言模型让这一目标重新变得可行。LLM 智能体把 shell 回显、扫描器输出与工具文档转化为规划上下文,即可用自由文本推理下一步动作,以极少的手工逻辑驱动一次多步入侵。沿用这一范式的系统,在真实主机与 CTF 风格主机上把侦察、利用与后渗透串联起来 [pentestgpt, hacksynth, rapidpen],一批不断增长的基准则度量这类智能体能走多远 [autopenbench, pentesteval]。其许诺是在不同产品与 CVE 之间实现广泛泛化,尽管现有系统在实践中仍需为每个目标做专门的命令落地,才能打响某个具体利用。

困难在于,渗透测试是一个稀疏信号下的长程决策过程。成功——一个 shell、一面 flag、一次 root——只在一条可能横跨数十个动作的链条末端到来,而其中大多数动作是悄无声息地走错的:重复的扫描、瞄错服务的利用、当前立足点根本支撑不了的后渗透动作。只按终端结果评判的智能体,得不到任何关于“中间哪一步走错了”的信号;近期的失败模式分析正是把智能体大量崩溃归因于这类逐步错误在长上下文中的累积 [excalibur]。文献的应对方式是在 LLM 外围加结构:对抗状态遗忘的任务树与推理模块 [pentestgpt]、规划器/摘要器回路 [hacksynth]、约束长程路径的外部经典规划器 [checkmate]、证据引导的攻击树搜索 [excalibur],以及为新智能体实例压缩先验知识的交接协议 [chap]。这些方法改进了智能体组织其轨迹的方式,但智能体在选择下一步动作时,仍然没有一个“该动作把杀伤链(kill chain)向前推进了多少”的学得估计。

对推理轨迹做步级评价,正是过程奖励模型(PRM)被提出来解决的问题。在数学与工具使用推理中,对每个中间步而非仅对最终答案打分,同时改善了验证与策略学习 [lightman, letsreward, prmsurvey],这一思路也已被引入交互式智能体 [agentprm, steca, qvalue]。把它用于渗透测试的障碍是监督信号。已有工作的过程标签,要么来自密集的人工逐步标注 [lightman],要么来自以采样后续来估计单步价值的蒙特卡洛/MCTS 回滚 [agentprm, qvalue],二者在攻防安全中都不可行:每一次回滚都是一次针对真实主机的真实利用尝试,单步真值在真实靶机上没有廉价的获取来源。因此,最强的学习型渗透智能体只能针对结果级或稀疏奖励进行优化。Pentest-R1 是其中与本文最接近的一个,它在 500 余条攻略与 CTF 任务上以“先离线后在线”的两阶段 RL 训练推理策略,但不含过程奖励模型;其奖励记在整条轨迹上而非单个决策上,因而无法定位数十步中究竟哪一步是错误,而这恰是失败模式工作所归因的那一类错误 [pentestr1, excalibur]。开放基准同样只评分结果或阶段级里程碑,而非逐个决策 [autopenbench, pentesteval]。另有一批工作直接应用强化学习,在网络模拟器与仿真器中学习基于价值的或分层的攻击策略 [nasimemu, hierdrl, hierexpert, dqn],甚至在真实环境上引导后渗透 [raiju];这些工作迁移的是一个用于行动的学得攻击策略,而过程奖励的场景需要的是一个学得的步级评价器,能为真实 LLM 智能体针对真实目标提议的动作打分。

把上述脉络放到两条坐标轴上——单步信号是否为学得的价值,以及领域是否为攻防渗透测试——会留下一个无人占据的格子。被提示的 CTF 评判能为攻击动作打分却什么也学不到;面向数学、工具使用与通用智能体的训练型 PRM 已经存在 [agentprm, steca, qvalue],但渗透测试领域没有;规划与框架方法确实给候选动作排了序,用的却是结构化动作集上人工设定的搜索启发式或符号规划器,而非对 LLM 实际提议的自由文本动作学得的价值。本文以一个 proposer 条件性的重排器占据这一格子:一个由模拟器训练得到、对真实 LLM 智能体已经收窄的候选集重排、并贯通一条完整真实杀伤链的 PRM。我们把 RL 智能体在抽象模拟器中学到的单步价值,当作一种无需任何真实靶机标注的过程标签来源,在这些标签上训练一个状态条件化的 PRM,再把它迁移到真实目标,用以重排真实 LLM 提议的动作,从而在不使用一条真实靶机标签、也不向评价器泄漏隐藏任务状态的前提下获得步级监督。

本文的贡献如下。

- 我们把自主渗透从终端成功信号提升为稠密的步级过程奖励:一个为每个候选下一步动作“将杀伤链向前推进多少”打分、并据此重排智能体动作的模型(Pentest-PRM)。据我们所知,这是首个面向攻防渗透测试的训练型步级过程奖励模型,区别于被提示的 CTF 评判,区别于面向防御性代码的训练型 PRM,也区别于面向数学、工具使用或通用智能体的过程奖励模型 [agentprm, steca, qvalue]。在真实 LLM proposer 已经收窄的候选集上,受重排的智能体有 52.7% 的步实现前向推进,而无辅助的 LLM 为 37.6%(+15.1pp,episode 聚类 p = 0.0012);但在完整动作面上,同一模型并不能胜过随机重排,这一 proposer 条件性的适用范围我们在第五节刻画。这是一项步级的过程收益,而非在这些单服务靶机上的结果改进。

- 我们在没有真实靶机标签的情况下获得该奖励。我们在一个抽象的单主机 Web MDP 中训练 DQN 价值 oracle,读取其单步价值作为候选动作的质量标签,在这些标签上训练状态条件化的 PRM,再经一道带安全门的 φ/ψ/η 适配器把它迁移到真实目标(φ:真实输出→抽象状态;ψ:LLM 文本→16 个冻结动作类型;η:动作→具体命令)。该流程以零真实靶机标注代价产出一个可用的过程奖励(它确实需要一次完整的 RL 价值 oracle 运行与一次种子门选择),且评价器输入不泄漏任何隐藏任务真值。ψ 解释器在有标注基准上达 95.5%、在更难的留出夹具上达 78.5%;泄漏审计确认重排输入不携带秘密信息,且在遮蔽上下文下表现稳健。

- 我们在整机上验证迁移后的奖励。通过重排真实 LLM 智能体的动作,它使智能体在无辅助时会卡住的、攻击模态各异的整机 VulnHub 靶机上完成贯通至 root 的真实杀伤链(DC-1:18/18 取根对 10/18;Symfonos:1:10/10 对 2/10),逐阶段拆分则把其价值定位在同机本地提权阶段——无辅助 LLM 能到达立足点却在此卡住。我们报告两台靶机;第三台 Toppo 暴露出重排器无法抬升的 proposer 天花板(两组均为 0%,因为利用动作从未被提议)。步级排序提升在三家 LLM 厂商上复现(top-1 prm > 仅 LLM 于 20/21 个厂商-靶机组合);结果级救援则出现在 proposer 尚有余量之处(如 Joomla,三家厂商皆然)。

这些结果带有两条局限,我们在引言中点明,并在第五、六节把它们咬合到最强的论断上。该 PRM 是一个 proposer 条件性的重排器,而非独立排序器:喂入原始动作面时它落到随机重排之下,且在这些单服务靶机上,其步级收益并不转化为更多 episode 达成完整目标。其价值标签也从模拟器奖励中继承了对侦察的乐观估值,这在抽象 MDP 中无害,但在真实目标上无用;第六节将其诊断为一个种子相关的“模拟到真实”奖励设计缺口,阶段拆分则把整机上的关键收益归于提权而非侦察。

本文余下部分组织如下。第二节将本文与 LLM 驱动的渗透测试、过程奖励模型以及面向自动化渗透的强化学习三条脉络对照定位。第三节形式化抽象单主机 Web MDP、价值 oracle 与 PRM。第四节描述无标签获取与带安全门的 φ/ψ/η 迁移适配器。第五节报告第一阶段评估、真实目标迁移、整机验证、跨厂商鲁棒性,以及侦察高估这一局限。第六节讨论局限,第七节总结全文。

## 相关工作

### A. LLM 驱动的渗透测试

LLM 智能体把渗透测试重构为对工具输出的自由文本推理,而其核心工程问题是在一条长而嘈杂的轨迹上维持连贯的计划。与本文新颖性最接近的是学得奖励这一脉络。Pentest-R1 以两阶段 RL 训练推理策略,先在 500 余条攻略上做离线模仿,再在 CTF 任务上做在线 RL [pentestr1];其奖励在轨迹级针对成功与里程碑结果计算,因而无法把一次失败归因到导致它的具体决策,而失败模式分析正把这一点判定为主导性的崩溃方式 [excalibur]。我们补上它所缺的部分——一个对候选动作学得的步级价值——并且无须一个结果级 RL 回路在真实目标上所必需的在线真实主机回滚就能获得它。

另有一批工作把全局结构外包给一个外部推理器,在此过程中也对候选动作施加了一种单步排序,但那是人工设定的。CHECKMATE 用经典规划器驱动智能体、约束长程路径 [checkmate];Excalibur 在其失败模式研究之上,增加了工具技能层、任务难度评估与证据引导的攻击树搜索 [excalibur]。它们的排序是结构化动作集上的搜索启发式或符号规划器给出的,而非对 LLM 实际提议的自由文本动作学得的价值,而后者正是我们提供的信号。其余系统着眼于轨迹连贯性而非单步打分:PentestGPT 引入任务树与推理模块,对抗长流程中的状态丢失 [pentestgpt];HackSynth 以规划器加摘要器构成双模块回路 [hacksynth];RapidPen 从单一 IP 起步,组合 ReAct 规划、成功案例检索与命令/执行反馈回路以获取 shell [rapidpen];CHAP 把先验知识压缩成交接协议,传给一个全新智能体实例以抑制上下文膨胀 [chap]。进展由那些评分结果或阶段级里程碑、而非逐个决策的基准来追踪:AutoPenBench 覆盖 33 个漏洞任务 [autopenbench],PentestEval 覆盖 346 个任务,对信息收集、弱点筛查、攻击决策与利用生成做阶段级评分 [pentesteval]。本文的贡献与上述工作正交:一个学得的步级过程奖励,用以重排 proposer 的动作,而非又一层规划、框架或基准。

### B. 过程奖励模型

过程奖励模型评分每个中间步的质量,而非只看最终答案。Lightman 等表明,步级的人工过程监督在数学推理上优于仅看结果的奖励,这也是 PRM 既支持训练、又支持测试期步级选择与重排的原因 [lightman];一条并行的工作把步级奖励模型与对推理路径的启发式贪心搜索耦合 [letsreward],一篇近期综述则梳理了数据生成、PRM 架构、测试期与 RL 用法的设计空间 [prmsurvey]。由于人工步标签昂贵,ThinkPRM 把验证器做成生成式的,从少量过程标签出发产出长链式的验证推理 [thinkprm];另有若干方法以采样估计过程价值:AgentPRM 用蒙特卡洛回滚为交互式智能体标注过程奖励,并以 actor-critic/RLHF 风格优化策略 [agentprm];STeCa 比较步级奖励以定位坏步,再用反思构造校准轨迹 [steca];步级 Q 值模型用 MCTS 标注步,并以 DPO 训练一个对动作排序的排序器 [qvalue]。这些方法的标签要么来自人工标注,要么来自对所评分任务本身的 MC/MCTS 回滚,两者都无法平移到攻防安全:每一次回滚都是一次真实利用尝试,单步真值在真实主机上没有廉价的获取来源。我们转而从一个在抽象模拟器中训练的 RL 智能体的价值函数读取过程标签。在安全场景中,最接近的奖励模型工作恰好分居于我们瞄准的格子两侧:被提示的评判能为攻击动作打分却什么也学不到,迄今的训练型 PRM 面向防御性代码而非攻击动作,因此没有一个是面向攻防渗透的、训练得到、可迁移、无标签的步级过程奖励。

### C. 面向自动化渗透的强化学习

另有一条脉络训练 RL 智能体充当攻击者。基于价值的深度 RL 提供骨干 [dqn],NASimEmu 在“模拟器训练—仿真器评估”(simulation-to-emulation)的设定下训练可泛化的网络攻击智能体,把学得的策略从模拟器迁移到一个仿真网络 [nasimemu]。为应对庞大的动作空间与长程视野,分层智能体把策略分解为高层与低层控制器,以比扁平 DQN 更快收敛 [hierdrl];以规则与知识图谱注入专家先验,则在一个两层智能体中进一步约束并建议动作 [hierexpert]。Raiju 超越模拟,用 A2C/PPO 选择动作、调用 Metasploit 模块,在真实环境上完成提权、抓取哈希与横向移动 [raiju]。在上述每一种情形中,训练所得的产物都是一个被部署去行动的攻击策略。本文对 RL 的用法在角色上不同:价值 oracle 是标签来源,而非被部署的智能体。我们读取其单步价值作为过程奖励目标,在这些目标上训练状态条件化的 PRM,再迁移这个评价器,用以评判并重排真实 LLM 智能体针对真实目标做出的动作。

## 参考文献

- [agentprm] S. Choudhury. "Process Reward Models for LLM Agents: Practical Framework and Directions." arXiv:2502.10325, 2025.
- [autopenbench] L. Gioacchini et al. "AutoPenBench: Benchmarking Generative Agents for Penetration Testing." arXiv:2410.03225, 2024.
- [chap] M. Vangeli, J. Brynielsson, M. Cohen, and F. Kamrani. "Context Relay for Long-Running Penetration-Testing Agents." NDSS Symposium Workshop (LAST-X), 2026.
- [checkmate] L. Wang et al. "Automated Penetration Testing with LLM Agents and Classical Planning." arXiv:2512.11143, 2025.
- [dqn] V. Mnih et al. "Human-level control through deep reinforcement learning." Nature, 518:529–533, 2015.
- [excalibur] G. Deng et al. "What Makes a Good LLM Agent for Real-world Penetration Testing?" arXiv:2602.17622, 2026.
- [hacksynth] L. Muzsai et al. "HackSynth: LLM Agent and Evaluation Framework for Autonomous Penetration Testing." arXiv:2412.01778, 2024.
- [hierdrl] K. Tran et al. "Deep Hierarchical Reinforcement Agents for Automated Penetration Testing." arXiv:2109.06449, 2021.
- [hierexpert] Q. Li et al. "A Hierarchical Deep Reinforcement Learning Model with Expert Prior Knowledge for Intelligent Penetration Testing." Computers & Security, 132:103358, 2023.
- [letsreward] Q. Ma et al. "Let's Reward Step by Step: Step-Level Reward Model as the Navigators for Reasoning." arXiv:2310.10080, 2023.
- [lightman] H. Lightman et al. "Let's Verify Step by Step." arXiv:2305.20050, 2023 (ICLR 2024).
- [nasimemu] J. Janisch et al. "NASimEmu: Network Attack Simulator & Emulator for Training Agents Generalizing to Novel Scenarios." arXiv:2305.17246, 2023.
- [pentesteval] R. Yang et al. "PentestEval: Benchmarking LLM-based Penetration Testing with Modular and Stage-Level Design." arXiv:2512.14233, 2025.
- [pentestgpt] G. Deng et al. "PentestGPT: An LLM-empowered Automatic Penetration Testing Tool." USENIX Security Symposium, 2024. arXiv:2308.06782.
- [pentestr1] H. Kong et al. "Pentest-R1: Towards Autonomous Penetration Testing Reasoning Optimized via Two-Stage Reinforcement Learning." arXiv:2508.07382, 2025.
- [prmsurvey] C. Zheng et al. "A Survey of Process Reward Models: From Outcome Signals to Process Supervisions for Large Language Models." arXiv:2510.08049, 2025.
- [qvalue] Y. Zhai et al. "Enhancing Decision-Making for LLM Agents via Step-Level Q-Value Models." Proc. AAAI Conference on Artificial Intelligence, 2025.
- [raiju] V.-H. Pham et al. "Raijū: Reinforcement Learning-Guided Post-Exploitation for Automating Security Assessment of Network Systems." Computer Networks, 2024. arXiv:2309.15518.
- [rapidpen] S. Nakatani et al. "RapidPen: Fully Automated IP-to-Shell Penetration Testing with LLM-based Agents." arXiv:2502.16730, 2025.
- [steca] H. Wang et al. "STeCa: Step-level Trajectory Calibration for LLM Agent Learning." arXiv:2502.14276, 2025.
- [thinkprm] M. Khalifa et al. "Process Reward Models That Think." arXiv:2504.16828, 2025.
