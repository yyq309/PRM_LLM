# 第一阶段报告：抽象单主机 Web 模拟环境、Web-RL 价值 Oracle 与 Pentest-PRM 的鲁棒性

> 范围：本报告只覆盖**训练阶段（抽象模拟器内）**——不含真实靶场 / adapter / 推理。所有数字取自
> `outputs/` 中的实跑结果（`training_stage_summary.json`、`coverage_audit.json`、`honest_eval.json`、
> `qstar_report*.json`、`prm_*_eval.json`、`reward_sensitivity.json`、`info_value_permissive.json`、
> `leakage_audit.json`、`label_confidence_report.json`、`hard_vs_loose.json` 等），未做任何美化。
> 复现：`python scripts/run_training_stage.py`（加 `--include-slow` 跑 oracle 重训 + Q* + MC）。
> 测试：`python -m pytest tests/ -m "not slow"` → **391 passed / 1 skipped**。

---

## 1. 概述

第一阶段在**纯抽象的单主机 Web 渗透 MDP** 内训练一个 **Web-RL 价值 Oracle**（DQN），用它产出
`Q_web(o,a)` / `V_web(o)` / `value_gap` 标签，再训练一个 **过程奖励模型 Pentest-PRM**（对 LLM 候选
动作打分/排序）。本报告论证三件事：

1. **模拟环境包含什么**（动作空间、观测、奖励、掩码、任务族与具体场景）；
2. **RL 价值 Oracle 的鲁棒性**（去饱和的诚实指标、Q\* 验证、多种子置信度、信息价值对齐、无泄漏）；
3. **PRM 的鲁棒性**（排序质量 + 置信区间、跨实例/跨链型泛化、校准、脏观测/OOD、错误动作识别）。

**贯穿原则：诚实优先于好看的指标。** 凡是被掩码/规则行/诱饵奖励抬高的数字，都标注并改报"增量"口径。

---

## 2. 模拟环境构成

### 2.1 冻结的 16 动作 schema（`web_attack_sim/action_space.py`）

| # | 动作 | # | 动作 |
|--:|---|--:|---|
| 1 | service_enumeration | 9 | vulnerability_check |
| 2 | http_fingerprint | 10 | exploit_attempt |
| 3 | web_path_enumeration | 11 | file_upload_attempt |
| 4 | content_retrieval | 12 | command_execution |
| 5 | input_discovery | 13 | sensitive_file_read |
| 6 | form_interaction | 14 | privilege_escalation |
| 7 | auth_attempt | 15 | post_exploitation |
| 8 | credential_use | 16 | stop_or_report |

### 2.2 观测 / 抽象状态（`AbstractWebState`，12 个可观测字段）

`open_services, tech_stack, discovered_paths, known_forms, known_parameters,
suspected_vulnerabilities, verified_vulnerabilities, credentials, auth_state, shell_state,
privilege_level, read_files`（外加 `remaining_budget`、`failed_actions`/`failed_branches`）。
**隐藏的任务真值（目标文件、口令、flag）不进入观测**，只通过反馈事件间接暴露——这是后面"无泄漏"
审计的前提。

### 2.3 奖励设计（`web_attack_sim/reward.py`，分级、信息论驱动）

每步基础 `step_cost = -0.1`，叠加**进展事件**正奖励与**错误事件**负奖励：

| 进展事件 | 奖励 | | 错误事件 | 惩罚 |
|---|--:|---|---|--:|
| service_found | +1.0 | | no_new_information | -0.3 |
| fingerprint_found | +1.5 | | duplicate_action | -1.0 |
| path_found / input_found | +2.0 | | credential_invalid | -1.0 |
| file_written | +3.0 | | vulnerability_not_present | -1.0 |
| credential_found / vuln_verified / cmd_exec | +4.0 | | precondition_missing / auth_required | -1.5 |
| session_obtained | +5.0 | | unsupported / unsafe / invalid_action | -3.0 |
| exploit_succeeded | +6.0 | | | |
| shell / sensitive_file_read / privesc | +8.0 | | **goal_reached** | **+12.0** |

奖励量级近似"信息增益"（拿到 shell/读到敏感文件比单纯枚举价值高一个量级），错误分级（重复→前置缺失
→越界/不安全）支撑 PRM 的**错误轨迹评分**（§4.6）。

### 2.4 动作掩码 + 前置条件守卫

环境对每个状态给出**合法动作掩码**（例如未拿到 shell 不能 `command_execution`），并有 precondition
guard。**注意（诚实口径）**：掩码本身就完成了大量"筛选"，所以"掩码下的 goal_rate"会饱和——后面所有
Oracle 指标都改报"在掩码内相对随机的增量"（见 §3.2）。

### 2.5 任务集与**具体场景**（`coverage_audit.json`）

**65 个任务、12 个结构族、12 个不同的拓扑签名**；每族 5 个实例（`injection_login` 10 个），全部带
2 条诱饵路径、全部有专家计划 + 轨迹。链深 2–12（直方图：深度 8 占 20 个，深度 7/9 各 10 个）。
难度：easy 10 / medium 15 / **hard 40**（hard mode 为 canonical，见 §3.5）。

12 个族即 12 类**具体攻击场景**（拓扑链 = 16 动作的有序组合）：

| 族 (难度) | 漏洞类 | 抽象动作链（场景） |
|---|---|---|
| `leak_file` (easy) | 配置/备份泄漏 | path_enum → content_retrieval |
| `default_pw` (easy) | 弱口令 | path_enum → input_discovery → auth_attempt → path_enum → file_read |
| `leak_login` (medium) | 泄漏→登录 | path_enum → content_retrieval → input_discovery → credential_use → file_read |
| `injection_login` (medium) | SQLi / LFI | path_enum → input_discovery → vuln_check → exploit → credential_use → file_read |
| `rce_shell` (hard) | RCE→webshell | path_enum → input_discovery → vuln_check → exploit → command_execution → file_read |
| `rce_privesc` (hard) | RCE→提权 | …command_execution → **privilege_escalation** → file_read |
| `upload_default_shell` (hard) | 弱口令→上传 | auth_attempt → input_discovery → file_upload → command_execution → file_read |
| `upload_default_privesc` (hard) | 上传→提权 | …file_upload → command_execution → privilege_escalation → file_read |
| `upload_leak_shell` (hard) | 泄漏→上传 | content_retrieval(泄凭据) → credential_use → file_upload → command_execution |
| `authed_injection` (hard) | 登录后注入 | auth_attempt → … → vuln_check → exploit → command_execution（漏洞**在登录后**） |
| `chained_exploit` (hard) | 链式利用 | vuln_check(v1)→exploit(v1→凭据)→**vuln_check(v2 需 v1)**→exploit(v2→shell) |
| `leak_authed_privesc` (hard) | 全链到 root | 泄凭据→credential_use→认证后 RCE→command_execution→privesc→file_read（链深 12） |

漏洞类覆盖：rce 25、弱口令 20、配置泄漏 15、文件上传 15、提权 15、sqli 10、lfi 5。覆盖网格填充率
0.524；**空白格（如 `lfi@easy`、`rce@medium`）是当前 schema 结构上不可表达的，已显式列为下一版 schema
扩展候选**（`coverage_audit.json` 的 `empty_cells`）——这是诚实上报而非遗漏。

### 2.6 与真实靶场的对应（`tasks/VULNHUB_CORRESPONDENCE.md`）

每个抽象族对应一类真实单主机 Web 攻击链（按 chain/漏洞族层面，非逐字节复制）：`default_pw`↔Mr-Robot、
`chained_exploit`/`leak_authed_privesc`↔DC-1/Raven、`rce_privesc`↔Kioptrix/DC 系列、`injection_login`
↔SQLi/LFI 靶等。这条对应在第二阶段被真实靶场经验性验证（见第二阶段报告）。

---

## 3. RL 价值 Oracle 的鲁棒性

### 3.1 训练设置

DQN，在**permissive/maskless**（训练时不喂动作掩码）下训练，80k 步 × 3 种子的 **seed-gate** 选 canonical
（`outputs/oracle_seed_gate.json` → `seed_gate_12fam_80k/seed_2`）。结构族**留出划分**：45 训练 / 10 留出实例 /
10 留出链型（`leak_login`、`rce_privesc` 整族留出），**留出链型与训练集链签名重叠 = 0**（防止泄漏式泛化）。

### 3.2 诚实指标（去饱和）

| 指标 | 值 | 说明 |
|---|--:|---|
| **专家动作 top-1 相对随机的增量** | **+0.127** | **头条指标**（expert_top1 0.463，掩码内随机基线之上） |
| permissive 无掩码 goal_rate | 0.40 | 无掩码随机仅 0.0125 → Oracle 确实学到了东西 |
| （caveat）掩码 goal_rate | ~0.67–0.70 | **饱和**：掩码内随机也有 0.544，top-3 增量仅 +0.048 → 不作头条 |

**结论**：Oracle 是"弱但真实、掩码依赖"的——头条用 **top-1 +0.127 增量** 和**无掩码 0.40 vs 随机 0.0125**，
不报饱和的掩码 goal_rate。这一诚实化来自一次对抗式复审（曾发现头条被掩码抬高）。

### 3.3 Q\* 价值迭代验证（`qstar_report_goal.json`）

用精确价值迭代 Q\* 作真值对照。**literal-reward Q\***（容许"挤诱饵奖励"）下 Oracle 看似弱（非退化
Spearman 为负）——这是 **decoy-milking 伪影**。换 **goal-aligned Q\***（仅目标 +1、无法挤奖励）后：

| goal-aligned Q\* | 值 |
|---|--:|
| 非退化 top-1 一致率 | **0.71** |
| 非退化 top-3 命中 | **0.98** |
| 非退化 mean Spearman | **+0.32**（从负**翻正**） |

即：剥离挤奖励伪影后，Oracle 对"朝目标的进展"排序是**合理的**，不是弱。（严格门仍因 per-decision-gap
0.40 > 0.25 标 False，但排序头条指标已说明问题。）

### 3.4 多种子标签置信度（`label_confidence_report.json`）

留出集上 3 种子的**逐决策 rank 一致率 0.972**、**完全一致比例 0.917**、平均置信 0.613 → 用作标签的
Oracle 排序在种子间**高度稳定**，PRM 标签可信。

### 3.5 环境改进 #1：hard mode（让决策"有后果"）→ canonical

发现最深的瓶颈是**环境太宽容**：loose 下强制动作后掩码贪心仍有 0.95 恢复率，只有 ~20% 同状态决策真正
影响结果，Oracle 价值与"实际回报"在"决策要紧处"的 Spearman ≈ **-0.005（≈随机）**。把预算收紧到
`plan_len+2`（hard mode）后：

| 决策要紧处 | loose | **hard（canonical）** |
|---|--:|--:|
| Oracle 价值 vs 实际回报 Spearman | -0.005 | **+0.373** |
| 决策相关分组占比 | 0.20 | **0.49（2.4×）** |
| Oracle 选中实际最优 | 0.46 | 0.57 |

hard mode 还**顺带消除了 decoy-milking**（紧预算下挤诱饵会错过目标），使 §3.3 的奖励整形（potential-based
shaping）**变得冗余**、无需重训。

### 3.6 奖励敏感性 / 信息价值对齐 / 无泄漏

- **奖励敏感性 / 信息价值**（`reward_sensitivity.json`、`info_value_permissive.json`）：Oracle 价值随
  "信息增益更高的进展事件"单调上升，行为不靠单一奖励常数硬凑（混淆项已在复审中修正）。
- **泄漏审计**（`leakage_audit.json`）：逐字段掩码只让 PRM 指标**平滑下降、无悬崖**（cliff 字段 = 0），
  且**无任何 secret token（路径/口令/flag）进入 PRM 输入**；**Oracle 的 q 值不作 PRM 特征**。→ 模型靠
  可观测上下文，而非泄漏的隐藏真值。

---

## 4. Pentest-PRM 的鲁棒性

**模型**：在结构化"状态+动作"特征上的 HistGradientBoosting（**绝不**用 Oracle q 值作输入）。canonical 持久化
`outputs/prm_strong.joblib`。所有指标在 **oracle 标注子集**上报（full-set 被规则行抬高，见 caveat）。

### 4.1 排序质量 + 置信区间（`prm_strong_eval.json`）

| 切片 | pairwise | rank |
|---|--:|--:|
| oracle_all | **0.890**（bootstrap CI95 [0.843, 0.937]） | 0.799 |
| 留出**实例** | **0.980** | 0.972 |
| 留出**链型** | **0.80** | 0.617 |

（变体对比：joint 神经 PRM 在 oracle_all 达 0.943、留出链型 0.925；baseline TF-IDF 仅 oracle 子集
0.790。canonical 用 GBT 的 0.890，第二阶段 live A/B 也用它。）

> **诚实 caveat**：full-set pairwise 0.93 被**规则行抬高**（73% 规则行 score=0、平凡可预测），**不作头条**；
> 只报 oracle 子集。

### 4.2 跨链型泛化与学习曲线

- **零样本到新族**（`prm_new_family_zeroshot.json`，训练时整族排除）：all-new pairwise **0.873**
  （authed_injection 0.968 / chained_exploit 0.826 / leak_authed_privesc 0.844），仅比同分布参考 0.890 低 ~0.02。
- **学习曲线**（`prm_learning_curve.json`）：留出链型 pairwise 在 **K≈4–5 个族即饱和**到 ~0.80 → 加同抽象
  的族/实例主要是锐化 rank/校准，不是新增能力天花板（天花板是抽象本身，第二阶段处理）。

### 4.3 校准（ECE + 后校准）

oracle_all ECE 0.166；**sigmoid/Platt 后校准**把最难的留出链型 ECE 从 0.155 **腰斩到 0.067**、oracle_all
0.101→0.075，但会**损害本就良好校准的留出实例**（0.049→0.149）→ **选择性应用**，不全局套。持久化的 strong
PRM 带 sigmoid 校准的 rank 头（`rank_calibrated`）。

### 4.4 鲁棒性：脏观测 + **留出腐蚀族 OOD**（`prm_robust_eval.json`）

robust 变体加"脏观测增广 + 置信加权"训练：clean pairwise 0.931、clean ECE 0.038；同族脏观测 +0.0495。
**OOD（在训练中从未见过的腐蚀族：重标/乱序/伪诱饵/预算抖动）**：robust 的 rank 下降更小（0.0153 vs
baseline 0.0164），但 pairwise 优势 **-0.0017**。→ **诚实结论：OOD 收益边际/混合**——robust 买到的是一点
rank 稳定性，不是 pairwise 优势。（单种子、无 CI，已标注。）

### 4.5 错误动作识别（`error_action_eval.json`）

PRM 能正确**标记不该做的动作**：

| 错误类 | recall |
|---|--:|
| precondition_missing / unsafe / outside_scope / schema_gap / ambiguous（硬约束错误） | **1.00** |
| low（合法但低价值的差选择） | 0.79（precision 0.81） |

oracle 子集 score ROC-AUC **0.893**。这正是奖励设计（分级负奖励）所瞄准的**错误轨迹评分**能力。

### 4.6 轨迹信用分配（`trajectory_credit_eval.json`）

在 n//2 处注入"提前抢目标"的脱轨步：per-step（PRM 分 vs 实际 return-to-go）Spearman **0.224**（66.7%
轨迹为正），脱轨步被打低 **0.34**。**但前瞻性"分叉门"较弱**：在决策分叉处优先正确动作仅 0.50（score）/
0.25（rank 头）。→ **诚实**：PRM 能事后识别坏步，但不是可靠的前瞻分叉门——这是从弱 Oracle 继承的局限，
修复属研究扩展（MC-return 辅助标签或更强 Oracle；MC-blend 已作并行诊断 `prm_strong_mcblend.joblib`，但
纯 MC 会破坏错误识别，故 canonical 仍用 DQN 标签）。

---

## 5. 诚实局限（不隐藏的负面结论）

1. **掩码 goal_rate 饱和**：掩码内随机即匹配 → 报 top-1 +0.127 增量，不报 raw goal/top-3。
2. **Oracle 弱且掩码依赖**：permissive 无掩码 goal 仅 0.40；需 80k 步避免欠训（45k 退化到 0.05）。
3. **PRM full-set 被规则行抬高**：只报 oracle 子集（strong 0.890 / 留出链 0.80）。
4. **闭环无自治增益**：把 strong PRM 接入闭环自治 rollout，permissive goal 0.00（与 baseline 同）——
   **PRM 是步骤排序器，不是独立策略**；一次贪心错选就让长 maskless 回合脱轨。其真实价值是逐决策排序
   质量（pairwise 0.89），闭环 rollout 不是它的合适测度。
5. **真实世界天花板是抽象本身**（≈49% 真实 LLM 动作在 16 schema 外）——这是第二阶段问题。

---

## 6. 结论

- **环境**：12 族 / 65 任务 / 12 拓扑的抽象单主机 Web MDP，16 动作冻结、12 维可观测状态、分级信息论奖励、
  hard mode 让决策有后果；与真实 VulnHub 攻击链有族级对应。
- **RL Oracle 鲁棒**（诚实口径）：top-1 +0.127 增量、无掩码 0.40 vs 随机 0.0125、goal-aligned Q\* top-1
  0.71 / Spearman 翻正 +0.32、多种子 rank 一致 0.972、决策要紧处 Spearman +0.373、**无隐藏真值泄漏**。
- **PRM 鲁棒**（诚实口径）：oracle_all pairwise 0.890 [CI 0.843–0.937]、留出实例 0.980、留出链型 0.80、
  零样本新族 0.873、错误动作识别 ROC 0.893（硬约束错误 recall 1.00）、校准可选择性腰斩 ECE。
- **如实报告了**所有饱和/规则行/OOD-边际/闭环-无增益的负面结论。

复现：`python scripts/run_training_stage.py`；测试 391 passed / 1 skipped。
