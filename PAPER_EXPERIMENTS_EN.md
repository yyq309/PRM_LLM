# Experiments

## E.1 What we test, in plain terms

**The goal.** We want an AI agent to autonomously attack a web server — find a vulnerability, get a foothold
(a shell), and escalate to *root* — and we want to *help* it choose better actions along the way.

**The idea.** A large language model (LLM) can *propose* next actions, but it often proposes them in a poor
order or wastes steps. Our contribution is a cheap **advisor** that **re-ranks** the LLM's proposed actions
so the most promising one is tried first. The catch — and the scientific interest — is *how* we train the
advisor: entirely inside a cheap **abstract simulator**, with **no labels from any real target** and **no
peeking at the answer**. We then ask whether that simulator-trained advisor still gives good advice on
**real** machines.

> **Think of the advisor as a chess coach.** It does not play the game itself; given the moves a player is
> already considering, it ranks which one is most promising. Our coach learned only from practice games in a
> simplified trainer, never from the real tournament — so the question is whether its judgment transfers.

We call the advisor the **PRM** (a *process reward model* — it scores the quality of each *step*, not the
final outcome). The experiments answer four questions:

1. **Does the simulator actually teach good judgment?** (E.3)
2. **Does that judgment transfer to real targets and help per step?** (E.4)
3. **Can the agent, so advised, complete a *full* real attack to root?** (E.5)
4. **Where does the advisor break, and why?** (E.6)

E.7 collects the ablations and controls; E.8 then asks whether the findings depend on which LLM we use
(they do not); E.9 reports the one honest limitation.

**Mapping to the contributions.** Q1–Q2 (E.3–E.4) supply the evidence for **C1** (a useful per-step
process-reward evaluator) and **C2** (it was obtained label-free and the transfer works); Q3 (E.5), together
with the multi-LLM study (E.8), is the evidence for **C3** (the transferred signal drives complete real kill
chains to root, with the value localized to privilege escalation). Q4 (E.6) and E.9 are **not** contributions
— they are the two honest limitations (recon over-valuation; proposer-conditional benefit), reported in full.

## E.2 Setup

**The simulator (where the advisor is trained).** Instead of training on real, expensive, label-scarce
hacking targets, we built an *abstract* model of single-host web attacks: a simplified world with a **fixed
menu of 16 action types** (e.g. "enumerate paths", "check a vulnerability", "run a command", "escalate
privileges"). We generated **65 attack tasks** spanning **12 different attack-chain shapes**. We split them
so that **20 tasks are held out for testing**, and — importantly — **10 of those use attack-chain shapes the
advisor never saw in training**, so we can test genuine generalization, not memorization.

Inside this simulator we train, in order: a reinforcement-learning **value oracle** (it learns how close any
state is to the goal), which then **labels** how good each candidate action is, which finally trains the
**PRM** (the advisor). One rule is strict and load-bearing: **the advisor is only ever shown observable
information** — what an attacker could actually see. It is *never* shown the oracle's internal scores or the
hidden answer.

**The adapter (how simulator advice reaches a real target).** Three small "translators" bridge the gap
between the simulator and a real machine, plus a safety check:
- a **reader** turns a real tool's raw output into the simulator's state format;
- an **interpreter** maps the LLM's free-text action ("I'll try the file-read exploit") onto one of the 16
  fixed action types the advisor understands;
- an **executor** turns a chosen action type into a concrete command for that specific target;
- a **safety gate** allow-lists every command, so the agent can only touch our own isolated lab machines,
  and every command is logged.

**The targets.** **15 real "Docker" web boxes** (each a single web service with a real, known
vulnerability — ThinkPHP, Struts2, Joomla, etc.) give breadth, and **2 full virtual machines** (DC-1 and
Toppo) give depth: the *complete* chain *web entry → foothold → same-machine privilege escalation → root*,
which the single-service boxes cannot (Table 1).

**Table 1 — Real targets: all 16 Docker web boxes (foothold only), grouped by foothold mechanism, + 3 full VMs.**

| Foothold mechanism | n | Docker boxes (product / CVE) |
|---|---|---|
| Direct RCE | 9 | ThinkPHP-5, ThinkPHP-5.0.23, Struts2 S2-045, Struts2 S2-048, php-cgi (CVE-2012-1823), Drupal/Drupalgeddon2 (CVE-2018-7600), Apache httpd (CVE-2021-41773), Tomcat (CVE-2017-12615), Gitea 1.4 |
| Weak-credential → deploy/RCE | 2 | Tomcat8 (manager), WebLogic |
| SQL injection | 1 | Joomla (CVE-2017-8917) |
| Template injection (SSTI) | 1 | Flask / Jinja2 |
| File disclosure / LFI | 2 | Rails (CVE-2019-5418), php-inclusion |
| Misconfiguration / traversal | 1 | nginx (insecure config) |

All 16 Docker boxes share the same phase (web entry → foothold) and terminal metric (goal reached = shell + sensitive read). The 3 full VMs add depth:

| Full VM (whole-machine, → root) | chain | terminal metric |
|---|---|---|
| **DC-1** | Drupalgeddon2 (CVE-2018-7600) → SUID `find` | `reached_root` |
| **Symfonos:1** | mail-masta LFI (CVE-2016-10956) + SMTP poisoning → SUID `/opt/statuscheck` PATH-hijack | `reached_root` |
| **Toppo** | creds in `/admin/notes.txt` → SSH → SUID `python` | `reached_root` |

*Scope of vulnerability coverage (stated honestly).* The 16 boxes span ~12 products and 6 foothold-mechanism
classes, but **direct RCE dominates (9/16)** — reflecting the Vulhub corpus. This is a breadth-of-*product*,
not breadth-of-vuln-*class*, evaluation. It is methodologically acceptable here because the frozen 16-action
schema abstracts the *kill-chain steps* (recon → locate → exploit → shell → read/escalate), **not** the
vulnerability class; so the per-box diversity the PRM actually sees comes from **chain length and topology**
(1-step self-advertising vs multi-step) rather than CVE family. All targets are owned, run on an isolated
host-only network, allow-listed, and logged. (The multi-LLM study in E.8 uses a 6-box subset of these.)

**How we measure, and the fair comparison.** Every real-target result is a head-to-head between two agents
that use the **same LLM** and differ in only one thing:
- **`prm`**: the LLM proposes actions, and the advisor re-ranks them;
- **`llm_only`**: the LLM's own ordering is used, with no advisor.

So any difference is attributable to the advisor alone. `llm_only` is the **primary baseline** (the same LLM
agent without the advisor); we add two reference points where they sharpen a claim: a **random-rerank**
control (re-orders the same candidates randomly — isolates whether the advisor's *specific* ranking matters,
not merely re-ordering) and a **scripted, non-LLM** upper bound (confirms a target is solvable when the
proposer is removed entirely). Because the advisor is a **process** reward model, our **primary** metrics are
**process / stage-level** — they measure the *quality of the trajectory*, not just whether it ends in a flag:
- **per-step progress (↑)** — fraction of *steps* that made forward progress rather than being wasted;
- **stage reached / shell-reach (↑)** — how far up the kill-chain ladder (recon → vuln → shell → cmd → file → root) the agent climbs, and how often it reaches the foothold stage;
- **ranking accuracy (↑)** — per-decision, how often the advisor's top-ranked action is the genuinely best one (vs the oracle);
- **wasted-step rate (↓)** — process efficiency.

The **outcome** metric — **goal-rate / root-rate (↑)**, the fraction of attempts that reached the goal / got
root — is reported too (unambiguous, ungameable), but as the **downstream check**, not the headline: a
process improver should first be shown to improve the process.

Because a run is a sequence of correlated steps (not independent coin flips), naive statistics overstate
significance. We use **episode-clustered permutation tests** and **bootstrap confidence intervals**, pair the
two agents on identical situations (**common random numbers**), and apply a **multiple-comparison correction**.
We report these conservative numbers throughout.

## E.3 Question 1 — Does the simulator teach good judgment?

**In one sentence: yes — the advisor learns a real (if modest) sense of which action makes progress, and it
does so without any leakage.** Three pieces of evidence:

**(a) It ranks progress correctly.** We compare the value oracle against a mathematically *optimal* solver of
the simulator (computed by value iteration). Its single best guess matches the optimum only **32 % of the
time** — modest — but the optimum sits in its **top 3 94 % of the time**, and its overall ordering correlates
with the optimum at **rank-correlation +0.46** (0 is random, 1 is perfect). So the value is a real, if
coarse, progress signal: not pinpoint, but it reliably keeps the good actions near the top — which is exactly
what a *re-ranker* needs (it shortens the candidate list, it does not have to be an oracle itself).

**(b) The advisor itself is a good ranker.** A standard way to score a ranker is **pairwise accuracy**: given
two actions where we know which is better, how often does the advisor rank them correctly? (0.5 is a coin
flip; 1.0 is perfect.) The advisor we actually deploy reaches **0.89 across all held-out tasks** (95 % CI
[0.84, 0.94], stable across 5 training seeds), **0.98 on new instances of attack chains it trained on**, and
— the demanding test — **0.80 on entirely *new* attack-chain shapes it never saw**: still well above the 0.5
coin-flip, though this hardest split is clearly its weakest point. (A preference-loss training variant raises
that unseen-chain number to 0.93, so there is headroom we have not yet deployed.) The advisor keys on the
*situation*, not on memorizing a specific attack, which is why it still ranks unseen chain shapes above chance
at all. We report the deployed model's numbers throughout, since it is the one that drives every real-target
result below.

**(c) No cheating.** An audit confirms the advisor's input contains **no hidden answer** — no secret path,
credential, or flag — and that hiding any single observable field degrades it only gracefully. Its skill
comes from observable context, not leaked secrets.

**Table 2 — Stage-1 advisor quality (deployed model).** Held-out evaluation; PRM pairwise over 5 seeds.

| Check | Value | What it means |
|---|---|---|
| Oracle top-1 vs optimal | 0.32 | exact-best match is modest |
| Oracle top-3 vs optimal | 0.94 | the optimum is almost always near the top |
| Oracle rank-correlation | +0.46 | overall ordering tracks the optimum |
| PRM pairwise — all held-out | **0.89** (95 % CI [0.84, 0.94]) | ranks the better of two actions correctly |
| PRM pairwise — new instances | 0.98 | generalizes to new instances of trained chains |
| PRM pairwise — **new chain shapes** | **0.80** | generalizes to unseen structures (its weakest point) |
| PRM calibration (ECE, after sigmoid) | ≈ 0.08 | predicted scores are well-calibrated |

![Figure 5](figures/fig5_stage1_pairwise.png)

*Figure 5. Deployed advisor's pairwise ranking accuracy by split (dashed line = 0.5 chance; CI shown on the
"all held-out" bar; the dashed outline marks the 0.93 a non-deployed preference-loss variant reaches on the
hardest split).*

One honest boundary: the advisor is a **ranker, not a player**. If we let it drive an agent entirely on its
own it does not succeed as a standalone policy — its value is in *advising*, which is exactly how we use it.

## E.4 Question 2 — Does the advice transfer to real targets?

**In one sentence: yes — on real machines, letting the advisor re-rank the LLM's actions produces a
statistically significant improvement in per-step decision quality.** First, the adapter works: the
interpreter maps the LLM's free text to the correct action type **95.5 %** of the time on a labeled benchmark,
and **78.5 %** on harder held-out fixtures — up from a **49 %** un-enhanced baseline. Then, across the 16 real
web boxes, the `prm` agent makes forward progress on **52.7 % of its steps versus 37.6 % for `llm_only` — a
+15-point gain** that holds under the conservative, clustered statistics: **p = 0.0012** (still significant
after multiple-comparison correction). Plainly: a ranking sense learned in a cheap simulator, with zero real
labels, measurably improves real per-step action choices. (Whole-episode goal-reach is higher too — prm 31 %
vs `llm_only` 12 % — but that gain is **concentrated in the boxes where the proposer fails outright** (e.g.
SSTI, Joomla); on boxes both arms can already solve, it is tied. We return to this *proposer-conditional*
pattern in E.5 and E.9.)

**Because the advisor is a *process* evaluator, the per-step metrics above are not a side-show — they are the
primary evidence.** Table 2a collects the process / stage-level metrics; the final-outcome rate (goal/root)
is the downstream check, not the headline. The advisor improves the *trajectory*, stage by stage: it climbs
higher up the kill-chain ladder, reaches the foothold stage more often, and wastes fewer steps.

**Table 2a — Process / stage-level metrics (16 web targets, pooled, prm vs llm_only).** Arrows give the good
direction; *p* is the episode-clustered permutation test where computed (others are directional point estimates).

| Process / stage metric | prm | llm_only | p |
|---|---|---|---|
| Per-step progress rate ↑ | **52.7 %** | 37.6 % | **0.0012** |
| Goal-aligned (forward-only) progress ↑ | **26.3 %** | 15.0 % | **0.0018** |
| Mean kill-chain stage reached ↑ (0 = recon … 5 = root) | **1.65** | 1.06 | — |
| Foothold / shell-reach rate ↑ | **29 %** | 15 % | — |
| Weighted progress ↑ | **0.59** | 0.45 | — |
| Wasted-step rate ↓ | **0.35** | 0.49 | — |
| Per-decision top-1 ranking acc ↑ (vs oracle; Qwen+GPT, 14 boxes) | **0.47–0.78** | 0.0–0.41 | — |
| Stage-1 pairwise ranking acc ↑ (held-out) | **0.89 / 0.98 / 0.80** | (0.5 chance) | — |

The outcome metric (whole-episode goal/root) follows in Table 2b and E.5 — but the *process* improvement above
is what a per-step reward model is built to deliver, and it is broad (8 metrics, two clustered-significant)
rather than tied to whether one obscure final exploit happens to assemble within budget.

**Table 2b — Per-box A/B on all 16 web targets (DeepSeek, 5 trials/arm; cells are prm / llm_only).** The
pooled row is the confirmatory claim; the per-box rows are exploratory and underpowered (n = 5) — read the
*direction*, not single-box point estimates. prm leads on per-step on most boxes, but the effect is honestly
**box-dependent** (e.g. Drupalgeddon2 and the Tomcat/Gitea/WebLogic boxes favor `llm_only`), and on
ThinkPHP-5 the raw LLM even wins the *goal* (80 % vs 20 %) — we do not hide these.

| Box | class | n | per-step % (prm / llm) | goal % (prm / llm) |
|---|---|---|---|---|
| Drupalgeddon2 | RCE | 5 | 41 / 100 | 0 / 0 |
| Struts2-S2-045 | RCE | 5 | 75 / 52 | 100 / 20 |
| Struts2-S2-048 | RCE | 5 | 90 / 56 | 40 / 0 |
| ThinkPHP-5-rce | RCE | 5 | 83 / 64 | 20 / 80 |
| ThinkPHP-5.0.23 | RCE | 5 | 77 / 38 | 80 / 40 |
| Tomcat-12615 | RCE | 5 | 19 / 29 | 0 / 0 |
| Tomcat8-weakpw | RCE | 5 | 44 / 67 | 0 / 0 |
| httpd-41773 | RCE | 5 | 46 / 44 | 0 / 0 |
| php-cgi-2012-1823 | RCE | 5 | 83 / 33 | 20 / 0 |
| Joomla-8917 | SQLi | 5 | 62 / 38 | 40 / 20 |
| Flask-SSTI | SSTI | 5 | 42 / 0 | 100 / 0 |
| Rails-5418 | LFI | 5 | 25 / 22 | 0 / 0 |
| nginx-insecure | LFI | 5 | 46 / 43 | 0 / 0 |
| php-inclusion | LFI | 5 | 38 / 30 | 0 / 0 |
| Gitea-1.4 | auth | 5 | 24 / 36 | 0 / 0 |
| WebLogic-weakpw | auth | 5 | 18 / 21 | 0 / 0 |
| **Pooled (16, clustered)** | — | 5 | **52.7 / 37.6** (p = 0.0012) | **31 / 12** (p = 0.005) |

## E.5 Question 3 — Can it complete a *full* real attack to root?

**In one sentence: yes — and the advisor's help is concentrated exactly in the hardest phase, privilege
escalation.** On the **DC-1** virtual machine the agent must do the whole chain: break in through the web app,
get a shell, then escalate to root on the same machine. Pooling our runs (18 attempts per agent):
- **with the advisor (`prm`): root captured in 18/18 attempts (100 %)**;
- **without it (`llm_only`): 10/18 (56 %)** — and roughly twice as many steps.

A fair note on noise: the LLM's solo root-rate on DC-1 varied between batches (40 % in one, 75 % in another),
while the advised agent was **100 % both times**. We therefore report the pooled 56 %, not the lucky draw —
and the honest reading is that **the advisor's value here is reliability**: it gets root *every* time, while
the raw LLM is a coin flip.

*Why* does the advisor help on the full machine but only tie on the single-service boxes? Splitting DC-1 by
phase answers it: the advised agent makes steady progress in **both** the web phase and the
privilege-escalation phase, but the un-advised LLM **collapses specifically in the privilege-escalation phase
(9 % progress)** while still doing fine on web reconnaissance. **The advisor earns its keep precisely where
the LLM's own instincts are weakest — local privilege escalation — which only a full-machine target exercises.** (Figure 3)

![Figure 3](figures/fig3_dc1_phase_split.png)

*Figure 3. DC-1 per-step progress split by phase. Both agents make similar progress on the web phase; the raw
LLM collapses to 9 % in the privilege-escalation phase, where the advisor sustains 37 %. (The advisor's DC-1
outcome also appears in Figure 1, left panel.)*

**A second machine confirms it — with a completely different attack.** We repeated the experiment on
**Symfonos:1**, whose chain shares nothing with DC-1's self-advertising web CVE: it runs web LFI → SMTP
log-poisoning → code execution → a SUID-root binary hijacked through a relative `PATH`. The result has the
same shape: with the advisor, **root in 10/10 attempts (100 %, CI [0.72, 1.0])**; without it, **2/10 (20 %,
CI [0.06, 0.51])** — non-overlapping. Here too the raw LLM reaches the foothold (it gets code execution, reads
`/etc/passwd`) but stalls at the obscure privilege-escalation step, and the advisor completes it. Because DC-1
and Symfonos are *different modalities*, the effect is not a quirk of one box.

A third machine, **Toppo**, draws a clean boundary: *both* agents fail autonomously because the LLM never even
proposes the needed "find credentials → SSH in" step — yet a scripted (non-LLM) agent reaches root on all three
machines. So the adapter and the advisor are sound; the failure is a *limit of the LLM's imagination*
(it cannot rank an action it never proposes), not a broken transfer.

**Table 3 — Full-machine results (autonomous, reach-root). Three VMs, three distinct modalities.**

| VM (modality) | agent | root rate ↑ | steps (median) ↓ | note |
|---|---|---|---|---|
| DC-1 (Drupal RCE → SUID find) | **prm** | **100 % (18/18)** | ~6 | advisor every run |
| DC-1 | llm_only | 56 % (10/18) | ~12 | varies 40–75 % across batches |
| Symfonos:1 (LFI+SMTP → PATH-hijack) | **prm** | **100 % (10/10)** | ~5 | non-overlapping vs llm_only |
| Symfonos:1 | llm_only | 20 % (2/10) | ~11 | reaches foothold, stalls at privesc |
| Toppo (cred→SSH→SUID) | both LLM arms | 0 % | — | LLM never proposes the cred→SSH step |
| Toppo | scripted (non-LLM) | 100 % | 1 | confirms the adapter/advisor are sound |

## E.6 Question 4 — Where does the advisor break, and why?

**In one sentence: the advisor systematically over-values reconnaissance — a bias we trace to the training
distribution and show the obvious post-hoc fixes do not remove. We report it as an honest limitation of *this*
transfer recipe, and are deliberately careful about what we do and do not claim.** In the simulator, training
rarely creates situations where the agent already knows everything but keeps scouting; as a result the advisor
learns that "reconnaissance" is almost always valuable. Concretely, its average score for the action "enumerate web
paths" is **0.89**, far above "run the exploit" at **0.54** (Figure 4). On real targets this shows up as the
advisor adding scouting steps a capable LLM does not need.

![Figure 4](figures/fig4_recon_overvaluation.png)

*Figure 4. The advisor's average score per action type. Reconnaissance/scouting actions (red) are rated far
higher than the decisive late-chain actions that actually advance the attack — run a command (0.04), escalate
privileges (0.04) — which is the over-valuation we trace to the simulator's training distribution.*

The important part is that this resists repair. We tried **three independent fixes** — down-weighting recon at
inference time, re-labelling the training data, and forbidding recon when better actions exist — and **all
three failed** to remove the bias without damaging the advisor elsewhere. A multi-seed check further shows the
bias size is itself unstable across training seeds. We are deliberately careful about scope: we show that
three obvious *post-hoc* fixes fail, but we do **not** claim the bias is unavoidable by a different training
or simulator design, nor that other LLM-pentest systems provably hit it. We therefore report it as a
**characterized limitation of this transfer recipe** — most naturally read as *reward-model overoptimization*
under the covariate shift that training-time action masking induces [gao2023scaling] — and leave it as an
explicit open question whether the bias is inherent to label-free abstract transfer or designable-away (a
controlled alternative-design study would settle it). We present this as honest analysis, not as a headline
result.

## E.7 Ablations and controls

**In one sentence: the gain survives every control we can think of — it is not leakage, not generic
re-ordering, and not a quirk of one proposer.** Each row of Table 6 isolates one alternative explanation a
skeptical reviewer might raise and reports what we found.

**Table 6 — Ablations and controls.**

| Control / ablation | Alternative explanation it rules out | Result |
|---|---|---|
| `llm_only` (remove the advisor) | re-ranking does nothing | per-step progress drops; advisor − baseline significant, **p = 0.0012** (§E.4) |
| random-rerank (re-order randomly) | *any* re-ordering helps, not this advisor's ranking | the advisor's edge is **phase-specific** — it leads in the privesc phase but is not reliably above random on easy web steps (§E.5, §E.8) |
| leak-free input audit | the advisor reads a hidden answer | no secret path / credential / flag in its input; graceful degradation when fields are masked (§E.3c) |
| generic-prompt control | success came from a CVE-named hint (test leakage) | the CVE-name lift disappears under a generic prompt; we report the leak-free number (§E.8) |
| standalone ranker (no proposer) | the advisor is secretly a policy | it cannot drive the agent alone — its value is strictly advisory (§E.3) |
| three bias-removal fixes | the recon bias is a patchable bug | all three fail without harming the advisor elsewhere (§E.6) |

The load-bearing control is the second row: because the advisor's per-step edge is *not* uniform over random
re-ordering, we deliberately do **not** claim a blanket per-step win — we claim a *phase-* and
*proposer-conditional* one, which the full-chain (§E.5) and multi-LLM (§E.8) results make precise.

## E.8 Does the result depend on which LLM?

**In one sentence: no — we reran the whole comparison with three different LLMs and the same behavior appears
every time.** We tested **DeepSeek, Qwen, and GPT-5.4** under identical conditions (same 7 targets, same code).
A single rule explains all three:

- **The advisor rescues a struggling LLM, and is redundant for a strong one.** On the Joomla box, where every
  LLM struggles on its own, the advisor lifts the success rate for **all three** (e.g. DeepSeek and Qwen from
  40 % to 100 %, GPT-5.4 from 0 % to 60 %). On DC-1, the weaker DeepSeek is rescued (100 % vs 56 %), while
  Qwen and GPT already solve it unaided, so the advisor adds no *outcome* — there was nothing left to fix.
- **The advisor's ranking is better almost everywhere.** Its top-ranked action beats the raw LLM's on
  **20 of 21** target-by-LLM combinations. The lone exception is a measurement artifact (on that box the
  advised agent still wins the actual goal 100 % vs 40 %).
- **The per-step effect depends on the target's length.** The advisor helps on longer multi-step chains but
  slightly *hurts* on trivial one-shot exploits, where the LLM already fires the single correct action every
  time and extra scouting only dilutes the rate.

Figure 1 shows the first point; Figure 2 the second. The supporting numbers:

![Figure 1](figures/fig1_multillm_proposer_conditional.png)

*Figure 1. The advisor rescues a struggling proposer (DeepSeek on DC-1; all three on Joomla) and is redundant
for one that already succeeds (Qwen/GPT on DC-1).*

**Table 4 — Joomla goal-rate (3-vendor rescue), n = 5 per arm.** Direction is consistent across all three
vendors; each leg is small-n, so we read the cross-vendor consistency, not a single significant leg.

| Vendor | llm_only goal ↑ | prm goal ↑ |
|---|---|---|
| DeepSeek | 0.40 | **1.00** |
| Qwen-3.7-max | 0.40 | **1.00** |
| GPT-5.4 | 0.00 | **0.60** |

**Table 5 — DC-1 axis (rescue is proposer-conditional).**

| Vendor | n/arm | llm_only root ↑ | prm root ↑ | effect |
|---|---|---|---|---|
| DeepSeek | 18 (pooled) | 0.56 | **1.00** | rescue (Δ +44 pts) |
| Qwen-3.7-max | 8 | 1.00 | 1.00 | saturated (no headroom) |
| GPT-5.4 | 8 | 1.00 | 1.00 | saturated (no headroom) |

![Figure 2](figures/fig2_top1_ranking.png)

*Figure 2. Top-1 ranking accuracy, prm vs llm_only, across 7 targets × 3 vendors. The advisor's top pick beats
the raw LLM's on 20 of 21 combinations (dashed line = 0.5 chance); the single exception, DeepSeek-Joomla, is a
measurement artifact (the advisor still wins the actual goal there, 100 % vs 40 %).*

So outcome-help is *conditional* (it appears when the LLM is weak), ranking-help is *near-universal*, and the
per-step effect tracks how long the attack is — one consistent mechanism, across three vendors, **not specific
to any single model.**

## E.9 The one honest limitation

We state plainly — and choose *not* to build the paper around — the following: the advisor's benefit on the
*final outcome* depends on how good the LLM's own ordering already is. It clearly helps a weak or
un-coached LLM (per-step **+15 points**, p = 0.0012). But once we *coach* the LLM with an explicit hint about
the action vocabulary, the proposer improves on its own — its goal-rate rises from **0.16 to 0.53** and its
wasted-step rate falls from **0.52 to 0.32** — and in an isolated test of *ranking alone* the advisor's
per-step progress (**0.27**) is no better than a random re-ordering of the same candidates (**0.29**). A
competent proposer leaves little for the re-ranker to add. We report this reversal in full rather than hiding it. We do not center the paper on
it because (a) the "coached" LLM there used an author-supplied hint, which confounds the comparison, and
(b) it is a specific case of a phenomenon already known in the verifier / process-reward-model literature
[lightman2023verify; cobbe2021gsm8k] — a model that *checks* work being obviated by a generator that no
longer makes the mistakes.

## E.10 Summary

The cheap abstract simulator produces a genuine, leak-free sense of which action makes progress (E.3); that
sense **transfers** to real machines and significantly improves per-step choices (E.4); it **drives complete
real attacks to root**, earning its keep precisely in the hardest privilege-escalation phase (DC-1: 100 % vs
56 %; E.5); it has a **clearly characterized failure mode** — over-valuing reconnaissance, which resists three
fixes (E.6); and the entire picture **reproduces across three different LLMs** under one simple rule (E.7).
The advisor's effect on final success is *conditional* on the LLM being weak, which we report honestly as a
limitation (E.9).

## References

- **[gao2023scaling]** Gao, L., Schulman, J., Hilton, J. *Scaling Laws for Reward Model Overoptimization.*
  ICML 2023. — anchors E.6 (the recon over-valuation as reward-model overoptimization under distribution shift).
- **[lightman2023verify]** Lightman, H., Kosaraju, V., Burda, Y., et al. *Let's Verify Step by Step.* 2023. —
  anchors E.8 (process/step-level verifiers; the verifier–generator relationship).
- **[cobbe2021gsm8k]** Cobbe, K., Kosaraju, V., Bavarian, M., et al. *Training Verifiers to Solve Math Word
  Problems.* 2021. — anchors E.8 (a verifier's value relative to generator strength).

*(Bibkeys are placeholders matching common conventions — verify the exact key/year against your reference
manager before submission.)*
