# 第二阶段报告：真实 VulnHub 推理阶段——靶场信息与实验结果

> 范围：第二阶段是**推理阶段**——把第一阶段冻结的抽象训练产物（ψ 归一化器 + `prm_strong.joblib` +
> Oracle 价值先验）通过 **φ/ψ/η adapter** 接到**真实单主机 Web 靶场**上，量化两件事：
> (1) 16 动作抽象**漏多少**（out-of-abstraction 率）；(2) PRM 重排序在真实靶上**有没有 uplift**。
> 安全：所有真实执行只针对**自有、隔离、本机回环（127.0.0.1）、可弃**的训练容器，全程经安全闸 +
> 审计；只跑只读命令（`id`/`whoami`/`cat /etc/passwd`）。所有数字取自 `outputs/stage2_*.json`。

---

## 1. 两阶段设计与本阶段定位

```
真实 Vulhub 容器  ──nmap/curl/sqlmap 真实输出──►  φ 解析 ─► AbstractWebState
   ▲                                                              │
   │ φ 再观测 ◄─ η 执行真实命令(受闸) ◄─ PRM 重排 ◄─ ψ 归一化 ◄─ LLM 提候选动作
```

- **φ（新建）**：真实工具输出 → 抽象状态；**ψ（复用一阶段并增强）**：LLM 动作文本 → 16 动作之一；
  **η（新建）**：抽象动作 → 具体命令（受安全闸控制）。
- **安全壳 `stage2/safety.py`**：`AuthorizationGate`（环境确认串 + `confirmed_isolated` + kill-switch）、
  靶作用域（仅私有/回环/.lab，公网拒绝）、命令白名单 + 破坏性 token 黑名单、JSONL 审计。
- 复现：`python -m stage2.preflight`（离线就绪自检）、`python -m stage2.aggregate_multibox`（多靶聚合）。

---

## 2. Phase 1：离线 φ/ψ 抽象差距测量（决定性、零执行）

在 7 个手工标注的 VulnHub 级走查 fixture（71 步、4 族，来自公开 write-up 结构、明确标注非实采）上离线
回放 φ/ψ，对照逐步手标真值：

| 指标 | 值 | 含义 |
|---|--:|---|
| **越界率 out-of-abstraction** | **8.5%** | 16 动作结构上覆盖 ~92% 真实步骤 → **不需扩 schema** |
| **ψ 归一化准确率（域内步骤，留出）** | **49% → 78.5%** | 真正瓶颈是 ψ，不是 schema（见下） |
| φ 状态还原召回 | 94.8% | 解析器够用 |

**关键发现（与预期相反）**：抽象覆盖很好（8.5% 越界），真正的墙是 **ψ 归一化器**——一阶段的关键词 ψ
对真实操作措辞只有 49% 准确率。在**独立 benchmark**（124 条意图，与 fixture 不相交）上调一个 Stage-2 本地
增强层后，在 7 个 fixture 上当**留出**测得 **78.5%**，误纳保持 0。越界的 4 个缺口词全是**非 web 原语**
（SSH 登录、离线破哈希、su 切换、二进制/SMB 溢出），本就在"单主机 web"范围外。决策门：8.5% ≪ 60% →
**不扩 schema，进入真实执行**。详见 `STAGE2_PHASE1_REPORT.md`。

---

## 3. 真实靶场信息（4 个 Vulhub 单主机 Web 容器）

全部经 daocloud 镜像 + 断点续传拉取（Docker Hub 直连 TLS/EOF 失败），绑 `127.0.0.1`，按靶定制 η recipe，
实测 RCE 成功（`stage2/live_smoke.py` 端到端验证）：

| 靶 | 框架 / 漏洞 | CVE | 端口 | 抽象族 | 工作载荷（η recipe，curl） | 自报家门 | 实测 RCE |
|---|---|---|--:|---|---|:--:|---|
| ThinkPHP 5-rce | ThinkPHP 5.0.x 框架 RCE | — | 8080 | rce_privesc | `invokefunction` 路由（`%5C` 转义反斜杠） | 是（页面含 "ThinkPHP"） | `uid=www-data` |
| ThinkPHP 5.0.23 | ThinkPHP 5.0.23 框架 RCE | — | 8084 | rce_privesc | `captcha _method`（与 5-rce **相反**） | 是 | `uid=www-data` |
| Struts2 S2-048 | Struts 2.3.32 OGNL RCE | CVE-2017-9791 | 8083 | rce_privesc | `saveGangster` OGNL（root 上下文） | 是（"Struts2 Showcase"） | **`uid=0(root)`** |
| php-cgi 2012-1823 | PHP-CGI 参数注入 RCE | CVE-2012-1823 | 8082 | rce_privesc | `?-d+auto_prepend_file=php://input` + body | **否**（只有 apache/php，无框架 banner） | `uid=www-data` |

**靶场调试中暴露并修掉的真实 sim-to-real bug**（合成 fixture 藏住的）：① `subprocess(text=True)` 用 GBK
解码真实响应崩溃 → 取字节 UTF-8/replace；② φ 把 HTML 里的 CSS 误读成凭据 → 跳过 CSS/HTML 行；③ curl 对
`[]` 做 globbing → 用 `-g`；④ ThinkPHP 5-rce 的 captcha 路由 404（与 5.0.23 相反）→ 按靶实测载荷；⑤ 注释型
η 模板（privesc/post-exploit）把 `#` 喂安全闸崩溃 → 作 no-op；⑥ OGNL 的 `{}` 撑破 `eta_command.format()` →
改 str.replace（花括号安全）；⑦ Struts2 响应 chunked 被 RCE 输出撑坏（IncompleteRead）→ curl 仍能抓取。

> Drupalgeddon2（CVE-2018-7600，8081）descriptor + recipe 已就绪，但 vulhub 镜像需先做标准 web 安装
> （无 drush，跳转 install.php），跳过——差 1 步可补。

---

## 4. 真实自治 A/B 实验设置

**完整自治回路**：DeepSeek V4 Pro（`deepseek-v4-pro`）按抽象观测提候选动作 → ψ 归一化 → **PRM 重排** →
η 渲染按靶真实命令 → 受闸 LiveExecutor 执行 → φ 重建状态 → 循环。每靶 **6 次/臂**、预算 14、温度 0.5
（LLM 随机性即预期变异）；**目标 = 拿到命令执行 + 读到一个敏感文件**。

- **A 臂 `prm`**：PRM 对候选重排序后取首选；
- **B 臂 `llm_only`**：取 LLM 候选原序首个（无 PRM 重排）。
- PRM 候选特征仍走**训练期 ψ**（冻结 PRM 的同分布），增强 ψ 只管动作映射/η（见一阶段耦合结论）。
- 命令：`python -m stage2.live_ab_trials --proposer llm --model deepseek-v4-pro --trials 6`。

---

## 5. 实验结果

### 5.1 管线验证（`live_smoke`，固定序列）

4 个靶都端到端打通：η→受闸 LiveExecutor→真实容器→φ。例：Struts2 达成 **命令执行（root）+ 读 /etc/passwd**；
php-cgi/ThinkPHP 达成命令执行（www-data）+ 读 /etc/passwd。安全闸全程生效（无授权一律 `PermissionError`，
公网/破坏性命令拒绝），全部审计到 `outputs/stage2_live_smoke_audit.jsonl`。

### 5.2 多靶自治 A/B（`stage2_multibox_aggregate.json`）

| 靶 | 框架 | 自报家门 | PRM 达成 | baseline 达成 | Δ |
|---|---|:--:|--:|--:|--:|
| ThinkPHP 5-rce | ThinkPHP | 是 | **6/6 = 100%** | 3/6 = 50% | +50pp |
| ThinkPHP 5.0.23 | ThinkPHP | 是 | 4/6 = 67% | 4/6 = 67% | **0（平）** |
| Struts2 S2-048 | Struts/Java | 是 | 5/6 = 83% | 4/6 = 67% | +16pp |
| php-cgi 2012-1823 | （无 banner） | 否 | 0/6 = 0% | 0/6 = 0% | 0 |

**自报家门 3 靶合并**：PRM **15/18 = 83.3%**（Wilson CI95 [0.61, 0.94]） vs baseline **11/18 = 61.1%**
（CI95 [0.39, 0.80]）；**Δ = +22.2pp，双比例 z = 1.49，p = 0.137（不显著）**。
全 4 靶合并：PRM 62.5% vs 45.8%，Δ +16.7pp，p = 0.247。

### 5.3 诚实判读（实验的真正结论）

1. **方向一致、从不更差**：PRM ≥ baseline 在**每个靶**上（+50, 0, +16, 0），合并 +22pp，重排从未"帮倒忙"。
2. **统计上不显著**：n=6/靶、p=0.137、所有 CI 重叠。**单靶 100% vs 50% 的抢眼结果没有复现**——同框架的
   ThinkPHP-5.0.23 是平局 → 那个头条**部分是运气**；真实效应是"小、正向、有噪声、欠功效"。
3. **以漏洞可识别为前提（重要边界）**：php-cgi 无框架 banner → LLM 提不出 exploit → 两臂都 0%。PRM 是
   **重排序器，无法给一个从未被提出的动作排序**。固定序列 `live_smoke` 仍能打穿 php-cgi（η recipe 对），
   只是**自治发现**做不到。**我们没有**往 φ 注入"php-cgi 有洞"来凑绿——那是泄题。

---

## 6. 结论与局限

**已验证（端到端，真实靶）**：两阶段管线、安全闸、按靶 exploit、单步真实 RCE（含 root）、单靶/多靶自治
A/B 全部跑通。抽象差距小（8.5%），ψ 瓶颈已显著缓解（49%→78.5% 留出）。

**实验结论**：抽象训练的 PRM 在真实靶上给出**温和、方向一致、但统计欠功效**的重排序收益（自报家门靶合并
+22pp，p=0.137），且**以 LLM 能识别并提出 exploit 为条件**。这是方法在真实靶上端到端跑通并**诚实交代效应
量**——不是漂亮的全胜。

**局限 / 下一步**：
- **样本量不足**：6 次/靶功效不够；要做到显著需**每靶 ~20 次 + 再加 2–3 个自报家门框架靶**
  （WordPress/Joomla/Spring）。
- **"无 banner"漏洞**（如 php-cgi）需给代理加**主动探测**动作——这是改进 proposer，不是 PRM 本身。
- **网络/基建**：Docker Hub 直连不稳，需走镜像 + 断点续传；DeepSeek 偶发 SSL EOF（已加每次试验容错）。
- PRM 训练期 ψ 与增强 ψ 的耦合：若要把增强 ψ 进 PRM 特征路径，需重训 PRM（一阶段重训）。

**产物索引**：`STAGE2_PLAN.md`（计划）、`STAGE2_PHASE1_REPORT.md`（离线测量）、`STAGE2_PHASE2_RUNBOOK.md`
（实打流程）、`STAGE2_LIVE_RESULTS.md`（实测全表）、`stage2/targets/*.json`（4 靶描述符）、
`outputs/stage2_multibox_aggregate.json`（聚合 + CI + 检验）、`outputs/stage2_ab_trials_*.json`（每靶原始）。
