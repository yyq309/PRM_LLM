# 第二阶段（推理）评测指标体系——超越成功率

> 动机：**per-episode 成功率（goal-reach rate）是信息量最低、最不敏感、最容易被混淆的单一指标**：
> (1) 二值、方差大，n=6 下几乎注定不显著；(2) 被**提议器（LLM）质量**和**靶机难度**混淆，无法隔离 PRM
> 这个**重排序器**真正的工作；(3) 把"部分进展"（拿到 shell 但没读文件）当成彻底失败，丢掉大量信号。
>
> 本文定义一套**多维、可隔离、高样本量**的推理阶段指标。实现见 `stage2/engagement.py`（逐步遥测）、
> `stage2/live_ab_trials.py`（每靶聚合 + CI）、`stage2/aggregate_multibox.py`（跨靶池化 + 双比例检验）。

---

## 维度 1 ── 过程层 / 重排序质量（**一个排序器的头条指标**）

PRM 的工作是**在每个决策点对候选动作排序**，不是端到端求解。所以应直接测"逐决策"质量：

| 指标 | 定义 | 为什么关键 |
|---|---|---|
| **逐步进展率 per-step progress rate** | P(被选动作使 φ 抽象状态产生新信息) = progress_steps / total_steps，PRM 臂 vs baseline 臂 | **高样本量**（每靶 ~6×14 步 → 跨靶数百步）→ 双比例检验**有功效**，而 per-episode 成功率没有 |
| **每步信息增益 fields-gained** | 每步新建的原子抽象事实数（新 set 成员 + 推进的 scalar）| "价值的量"，比二值 progress 更细 |
| **去重生产动作数 distinct-productive-actions** | 产生进展的不同动作类型数 | 区分"真推进"与"在少数动作上空转" |

> 这是把 **per-episode 二值（欠功效）** 升级成 **per-step 比率（高 N、有功效）** 的关键——直接回答
> "跟着 PRM 的首选走,是否比跟着提议器原序走、更经常地推进"。

## 维度 2 ── 里程碑 / 部分信用（不是 all-or-nothing）

| 指标 | 定义 |
|---|---|
| shell 到达率 / **命令执行到达率** / 文件读取率 / **root 到达率** | 沿抽象攻击链的**分级**到达比例 |

一次"拿到 shell 但没读文件"在成功率里是 0，在里程碑里是"到达 shell/命令执行"——保留了真实进展信号。

## 维度 3 ── 效率

| 指标 | 定义 |
|---|---|
| 达标步数 steps-to-goal | 达标时用的步数（越少越好） |
| 浪费率 wasted-rate | 无进展步 / 总步 |

## 维度 4 ── 成本（真实交战才关心）

| 指标 | 定义 |
|---|---|
| 提议器调用数 proposer_calls | 每回合 LLM 调用次数（token/费用代理） |
| η 真实执行数 eta_executions | 每回合对真实靶发起的命令数（攻击足迹 / 噪声） |

## 维度 5 ── 抽象覆盖（实时 Phase-1）

| 指标 | 定义 |
|---|---|
| **实时越界率 live out-of-abstraction rate** | LLM 提议中 ψ 无法映射成 16 动作之一的比例 | 

这是 Phase-1 离线 8.5% 的**在线真实版**——在真实自治回路里，LLM 实际提出多少"框架外"的东西。

## 维度 6 ── 安全

| 指标 | 定义 |
|---|---|
| 安全闸拒绝数 gate_refusals | 被 `safety.AuthorizationGate` 拦下的命令数（越界/破坏性/不安全） |

安全闸**始终生效**——这本身是一个"零越权/零破坏"的安全性证据，应作为指标报告。

## 维度 7 ── 稳健性 / 统计

| 指标 | 定义 |
|---|---|
| 跨试验方差 | goal / per-step-progress 在多次试验间的稳定性 |
| 置信区间 + 显著性 | 所有比率给 **Wilson CI**；PRM vs baseline 给**双比例 z 检验 p 值** |

---

## 论文中应如何报告（建议）

1. **头条用维度 1 的"逐步进展率"+ CI + p**，不是 per-episode 成功率——因为它高 N、有功效、隔离 PRM。
2. **成功率降级为维度 2 的一个里程碑**（与 shell/命令执行/文件读取/root 并列），并诚实标注其 n 小、欠功效。
3. **同时报维度 4–6 的成本/越界/安全**——让评审看到的不只是"赢没赢"，而是"代价多大、抽象漏多少、是否安全"。
4. **每个数字都带 oracle 子集口径 / CI / 是否显著**，与第一阶段诚实口径一致。

一句话：**用"PRM 是否更经常地把对的下一步排在前面（逐步进展率，带 p）"作为主张，而不是"它是否打穿了靶（成功率）"**——前者是 PRM 真正负责的事、且统计上站得住；后者欠功效且被提议器/难度混淆。

---

## 复现

```powershell
# 每靶 6 次试验、真实 DeepSeek 提议、PRM 重排，输出全部维度 + CI
python -m stage2.live_ab_trials --target stage2\targets\<box>.json --proposer llm `
    --model deepseek-v4-pro --trials 6 --budget 14 --confirmed-isolated --report-output outputs\<box>.json
# 跨靶池化 + 双比例检验（per-episode 成功率 与 高-N 逐步进展率 两套）
python -m stage2.aggregate_multibox
```
