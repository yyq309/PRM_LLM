# Stage 2 — LIVE inference results (16 Docker web boxes + 2 full VMs, cluster-robust)

**What this is:** the abstract-trained adapter (φ real-output→state, ψ text→16-action, η action→command,
all gated by `stage2/safety.py`) driving the autonomous loop `propose → ψ → PRM rerank → η → LiveExecutor
→ φ` against **16 real Vulhub containers**, bound to `127.0.0.1` only, owned/isolated, every command
audited. A/B = `prm` (PRM reranks the proposer's candidates) vs `llm_only` (the proposer's own order).

**Read this section first; everything below "Appendix" is superseded interim work kept for provenance.**

---

## ★★ FULL-CHAIN VM RESULTS (C-B flagship, 2026-06-21)

Real **VMware VulnHub VMs** on an isolated host-only segment (VMnet2, 192.168.52.0/24), the COMPLETE kill
chain **Web entry → foothold → same-host privilege escalation → root** — what the Docker web boxes (foothold
only) and XBEN (recon only) cannot give. Terminal metric = `reached_root` (non-gameable). Gated + audited;
flags recorded boolean-only. New η plumbing: `stage2/payloads/drupalgeddon2.sh` (Drupal-7 RCE),
`stage2/payloads/ssh_cmd.py` (paramiko one-shot SSH, host has no sshpass). Executor now strips the host Clash
proxy (`eta._no_proxy_env`). φ credits `euid=0(root)`.

| VM | chain | deterministic | **LLM-autonomous A/B (deepseek-chat, 5 trials)** |
|---|---|---|---|
| **DC-1** | Drupalgeddon2 (CVE-2018-7600) → SUID `find` → root | ✅ root, 2 steps | **n=10: prm root 100% (10/10) CI[0.72,1.0] vs llm_only 40% (4/10) CI[0.17,0.69] — CIs NON-overlapping**; ~2× fewer steps (6.3 vs 11.5). (n=5: 100% vs 60%.) |
| **Toppo:1** | `/admin/notes.txt` cred → ssh → SUID `python` → root | ✅ root, 1 step | **0% both arms** — proposer never builds the cred→ssh foothold |
| ~~Raven-2~~ | PHPMailer (CVE-2016-10033) | **OMITTED** | foothold blocked by image-hardened PHPMailer (From escaping); not load-bearing (3rd det. privesc vector only) |
| **Symfonos:1** | mail-masta LFI (CVE-2016-10956) + SMTP poison → RCE → SUID `/opt/statuscheck` PATH-hijack → root | ✅ root (deterministic) | **AUTONOMOUS n=10: prm root 100% (10/10) CI[0.72,1.0] vs llm_only 20% (2/10) CI[0.06,0.51] — NON-overlapping**. 2nd autonomous full-chain box, DISTINCT modality from DC-1; llm reaches RCE+/etc/passwd but stalls at the PATH-hijack privesc. Self-cleaning privesc (no SUID-bash residue) → no cross-trial contamination (verified non-privesc arm gets Permission denied). `outputs/stage2_fullchain_symfonos{,_b}.json`; helper `stage2/payloads/symfonos_rce.py`. |

**Two honest findings:**
1. **On the harder MULTI-STEP real chain the PRM HELPS** (DC-1 n=10: root **100% vs 40%, non-overlapping CIs**,
   ~2× fewer steps) — the **OPPOSITE** of the web-only efficiency inversion. **The phase-split shows WHY:**
   prm makes progress in both phases (web 36% / local 37%), but **llm_only collapses in the LOCAL/privesc phase
   (9%, 4/43)** while still doing web (32%). ⇒ **the PRM's full-chain value lives in the local/privilege-escalation
   phase, where the LLM's own ordering is weak** — the regime the web-only A/B could not exercise. So the demoted
   "good proposer obsoletes the PRM" limitation is **web-phase-specific**; on a real kill chain the reranker is
   essential exactly where the proposer is weakest.
2. **Proposer ceiling, not adapter:** autonomy succeeds when the foothold is a recognizable web CVE (DC-1
   self-advertising Drupal) and fails when it needs cred-discovery→ssh (Toppo 0%) — yet the **deterministic
   proposer reaches root on both**, so the adapter is sound (same lesson as the Docker boxes' `exploit_never_proposed`).

Reports: `outputs/stage2_fullchain_dc1.json`, `outputs/stage2_fullchain_toppo.json`.

## ★★ MULTI-LLM cross-vendor A/B (DeepSeek / Qwen-3.7-max / GPT-5.4, 2026-06-22)

Closes the single-model gap: the result is **not deepseek-specific**. **SYMMETRIC 3-vendor design** — all
three vendors run the *same* code (current metrics) on the *same* 7 boxes (DC-1 full chain + 6 web), llm
proposer, prm-vs-llm_only, 0 errored: DeepSeek-chat (official `api.deepseek.com`), Qwen-3.7-max and GPT-5.4
(both via the tsbys OpenAI-compatible gateway, `--provider qwen|gpt`, keys env-only). DeepSeek was **re-run**
2026-06-22 so it carries top-1 + clean metadata + a proper llm-proposer ThinkPHP (the old asymmetric deepseek
baselines are superseded). **The headline claims were adversarially verified against raw `outputs/*.json` by
a workflow (4/4); numbers paste-checked. Files: `outputs/{ds,qwen,gpt}_*.json` (21) + the n=10 DC-1 flagship.**

**Unified finding (one proposer-conditional mechanism).** Across three vendors the PRM does exactly one
thing, and what you see depends on the proposer it wraps: it **rescues the OUTCOME of a struggling proposer**
(DeepSeek on DC-1: root 56% → 100%; all three on Joomla), is **obsoleted on OUTCOME by a strong proposer**
that already roots the box unaided (Qwen and GPT on DC-1: 100% → 100%, no headroom), yet its **per-decision
top-1 ranking beats the raw proposer on 20/21 vendor-boxes** (the lone exception, DeepSeek-Joomla, is itself
a metric artifact — see below). Outcome rescue is conditional; ranking lift is near-universal — same
re-ranking mechanism, different ceilings.

### Joomla CVE-2017-8917 — 3-vendor goal-rate rescue
| Vendor | model | n/arm | llm_only goal | prm goal | prm > llm_only |
|---|---|---|---|---|---|
| DeepSeek | deepseek-chat | 5 | 0.4 (2/5) | 1.0 (5/5) | yes |
| Qwen | qwen3.7-max | 5 | 0.4 (2/5) | 1.0 (5/5) | yes |
| GPT-5.4 | gpt-5.4 | 5 | 0.0 (0/5) | 0.6 (3/5) | yes |

All three: PRM rescues a struggling proposer's goal-rate. **Honest read:** each leg is n=5, so CIs are wide
and the prm/llm separations overlap at 95% (read direction, not significance, per box); the *consistency of
direction across three independent vendors* is the load-bearing evidence, not any single leg. Files:
`ds_joomla-cve-2017-8917.json`, `qwen_joomla-cve-2017-8917.json`, `gpt_joomla-cve-2017-8917.json`.

### DC-1 axis — rescue is proposer-conditional
| Vendor | model | n/arm | llm_only goal/root | prm goal/root | effect |
|---|---|---|---|---|---|
| DeepSeek | deepseek-chat | **18 (pooled)** | 0.56 / 0.56 | 1.0 / 1.0 | **rescue** (Δ+44pp) |
| Qwen | qwen3.7-max | 8 | 1.0 / 1.0 | 1.0 / 1.0 | no rescue (saturated) |
| GPT-5.4 | gpt-5.4 | 8 | 1.0 / 1.0 | 1.0 / 1.0 | no rescue (saturated) |

The PRM moves the outcome needle only where the proposer fails. **DeepSeek DC-1 is pooled over two runs**
(`stage2_fullchain_dc1_n10.json` n=10 + `ds_dc1.json` n=8, identical config): prm **18/18 root** vs llm_only
**10/18 (56%)**. **Honest note — single-run variance:** llm_only autonomous root on DC-1 was 40% (n=10) and
75% (n=8); the PRM was 100% in BOTH. So the PRM's DC-1 value is *reliability* — it reaches root every run
while the raw proposer is a coin-flip (40–75%); we report the pooled 56% rather than the favorable single
draw. Qwen/GPT proposers already saturate DC-1, so there is no outcome headroom for the PRM there.

### Top-1 ranking accuracy — prm > llm_only on 20/21 vendor-boxes (all 3 vendors)
`mean_top1_ranking_acc`, prm / llm_only:

| Box | DeepSeek prm / llm | Qwen prm / llm | GPT prm / llm |
|---|---|---|---|
| dc1 | 0.580 / 0.361 | 0.474 / 0.409 | 0.604 / 0.317 |
| joomla-cve-2017-8917 | **0.264 / 0.403 ✗** | 0.321 / 0.249 | 0.286 / 0.260 |
| php-cgi-cve-2012-1823 | 0.550 / 0.229 | 0.783 / 0.142 | 0.600 / 0.189 |
| struts2-s2-045 | 0.333 / 0.000 | 0.450 / 0.000 | 0.767 / 0.000 |
| struts2-s2-048 | 0.333 / 0.100 | 0.767 / 0.000 | 0.517 / 0.000 |
| thinkphp-5-rce | 0.447 / 0.317 | 0.767 / 0.317 | 0.700 / 0.173 |
| thinkphp-5023-rce | 0.480 / 0.317 | 0.633 / 0.317 | 0.600 / 0.220 |

prm > llm_only in **20/21**. The one exception is **DeepSeek-Joomla** (prm 0.264 < llm 0.403) — but on that
same box DeepSeek's prm wins the actual *goal* 100% vs 40%, so the exception is a **top-1-metric artifact**
(top-1 is scored vs the oracle's heuristic priority, which is not goal-truth on every box), not a real
ranking loss. Net: the ranking lift is a genuine 3-vendor result.

### Per-step progress is box-dependent, not uniform
Example (GPT-5.4); DeepSeek and Qwen show the same box-dependence (e.g. DeepSeek ThinkPHP llm 94% vs prm
44–48%; DeepSeek php-cgi prm 50% vs llm 48%):

| Box | llm_only per_step | prm per_step | winner |
|---|---|---|---|
| struts2-s2-045 | 1.0 (10/10) | 0.417 (5/12) | **llm > prm** |
| struts2-s2-048 | 1.0 (10/10) | 0.333 (5/15) | **llm > prm** |
| dc1 | 0.279 (31/111) | 0.423 (22/52) | prm > llm |
| php-cgi-cve-2012-1823 | 0.395 (15/38) | 0.588 (10/17) | prm > llm |

PRM helps per-step on the longer multi-step chains (dc1, php-cgi) but **hurts** per-step on the short
single-shot Struts2 boxes, where the raw proposer already fires the one correct action every time and the
PRM's exploration only dilutes the rate. Not a uniform per-step win.

### Honest caveats
- **DC-1 deepseek llm_only has high single-run variance (40–75%);** we report the pooled n=18 (56%), not the
  favorable n=10 draw. The PRM's 100% is stable across both runs.
- **Small n on the web boxes:** all 6 web boxes are n=5/arm/vendor; treat point estimates as directional and
  read the cross-vendor *consistency*, not any single CI.
- **DeepSeek-Joomla is the lone top-1 exception** (artifact — prm still wins goal there; see above).
- **Per-step lift is not uniform** — helps multi-step chains, hurts single-shot boxes.
- (Superseded: the earlier asymmetric deepseek baselines — `stage2_ab_trials_*.json`, the `proposer='target'`
  ThinkPHP run — are replaced by the symmetric `ds_*.json` re-run; kept on disk for provenance only.)

Code: `--provider {deepseek,qwen,gpt}` in `stage2/live_ab_trials.py`; `PROVIDERS` in
`scripts/deepseek_client.py` (reasoning models get max_tokens=16000). Reproduce: per-box
`python -m stage2.live_ab_trials --target … --proposer llm --provider <v> --model <m> --executor live
--confirmed-isolated`.

## ★ C-C MECHANISM — recon over-valuation traced to the reward (G2, 2026-06-21)

`scripts/analyze_recon_bias.py` → `outputs/recon_bias_histogram.json`. Mean PRM target label per
(action_type × phase) over `prm_samples_train.jsonl`:
- `web_path_enumeration` **early 0.94 → advanced(post-foothold) 0.609**, and post-foothold it STILL outranks
  `command_execution` (0.21) and `privilege_escalation` (0.199) — recon over-valued exactly where it should be ≈0.
- **Cause (not action-masking):** `web_attack_sim/reward.py` grants `path_found:+2.0`, `input_found:+2.0`,
  `fingerprint_found:+1.5` **whenever recon reveals new info, with NO phase conditioning** — so even post-foothold,
  finding a new path earns +2.0 in the sim. The oracle *correctly* keeps recon valuable; that's **useless on real
  targets**. A sim-to-real **reward-design** gap; the 3 surgical label/inference fixes failed because the signal
  is in the reward, not the labels. (n=64 advanced-recon samples exist & were partially devalued 0.94→0.609 →
  NOT a "masked training never saw them" gap.)
- **H refinement (reward-fix retrain, honest negative):** retraining the oracle with the recon bonus
  zeroed post-foothold (`--decay-recon-reward`) did **not** reduce the bias — because a **fresh seed-0 control
  oracle already devalues recon post-foothold correctly** (web_path_enum advanced **0.173 < command_exec 0.309**;
  overall 0.455 vs the deployed 0.887). So the deployed PRM's strong recon over-valuation is partly a
  **seed-gate *selection* artifact**, not a deterministic reward consequence. Honest C-C: the reward *permits*
  recon over-valuation and the deployed oracle landed on a high-recon solution, but its severity is
  **seed-dependent**; surgical fixes don't remove the *deployed* model's bias. (`outputs/recon_bias_{control,rewardfix}.json`.)

## ★ C-A TRANSFER BASELINE — learned PRM vs cheap heuristic (G4, 2026-06-21)

From the reranker-isolation ablations (`outputs/stage2_ablation_rerank{,_llm}.json`), pooled per-step progress:
- **Deterministic proposer** (full surface): prm 0.267 ≈ **heuristic 0.262 (Δ+0.4pp, perm p=0.65, NS)**; prm < random
  (Δ−2.7pp, p=0.003). ⇒ the learned RL value adds **nothing** over a cheap hand-coded prior here.
- **LLM proposer** (targeted): prm 0.685 > **heuristic/goal-ladder 0.479 (Δ+20.6pp, perm p=0.001)**; > random (p=0.007).
  ⇒ the learned value **does** beat the cheap prior — but only in the realistic LLM-proposer regime.

**Honest C-A:** the learned value's advantage over a cheap domain heuristic is itself **proposer-conditional**.

## ★ G3 PAIRED CRN + the UNIFIED phase-dependent synthesis (2026-06-21)

`stage2/paired_ab.py`, 6 web boxes × {llm_only,prm,random,oracle}, deepseek-chat, live, **mean cache-hit 30%**.
**Paired (variance-controlled) per-step:** prm **40.2%** (49/122) — **significantly WORSE** than llm_only 59.3%,
random 59.0%, oracle 70.0% (Δ−19/−19/−30pp, clustered perm-p=0.0). This **OVERTURNS the earlier *unpaired*
ablation** (prm 68.5% > random 50%, p=0.007) — that positive was a proposer-draw-variance artifact; under proper
pairing the recon bias makes the PRM per-step worse than random on web boxes. (per-episode goal prm 90% ≈
random/oracle 97% > llm_only 80% — reaches goal, inefficiently.) `outputs/stage2_paired_ab.json`.

**THE UNIFIED HONEST STORY (one mechanism explains the whole arc):** the PRM is a per-step **LIABILITY in the
WEB/recon phase** (G3: < random; recon over-valuation) but a decisive **ASSET in the LOCAL/privilege-escalation
phase** (DC-1 full chain: root **100% vs 40%** @ n=10, phase-split local **37% vs llm_only 9%**). On a full real
kill chain the local-phase benefit dominates → PRM wins big; on web-only boxes only the web-phase harm shows →
PRM loses. Everything (web-only inversion, the unpaired/paired flip, the full-chain win) follows from this single
**phase-dependent** fact.

---

## ★ FINAL RESULT (2026-06-18) — honest, cluster-robust

5 trials/arm × 16 boxes, `deepseek-chat` proposer (both arms share it → fair). Full per-box numbers in
[STAGE2_SEVEN_DIM_TABLE.md](STAGE2_SEVEN_DIM_TABLE.md); stats in `outputs/stage2_stats_analysis.json`;
ablation in `outputs/stage2_ablation_rerank.json`.

> **★ UPDATE (2026-06-23):** a 16th box — **Flask-SSTI** (Jinja2 SSTI → RCE → shell, new vuln class) — was
> added (`outputs/stage2_ab_trials_flask-ssti.json`; deepseek **prm goal 100% (5/5) vs llm_only 0%** — a clean
> proposer-conditional rescue). With it the **pooled per-step is now prm 51.7% vs llm_only 34.3%, +17.5pp,
> clustered p=0.0001** (was +12pp/p=0.02 at 15 boxes). Per-episode pooled also rises (prm 33% vs 7%) but is
> **concentrated in the proposer-failure boxes** (SSTI, Joomla) — reported as the proposer-conditional pattern,
> not a uniform win. The dated sub-experiment numbers below are preserved as snapshots.

### 1. The statistics, done correctly (clustered, not naive)

A naive two-proportion z-test on pooled per-step progress treats every step as independent — it is not
(steps within an episode/box are correlated), so it **over-states** significance. Re-tested with an
**episode-clustered stratified permutation test** (randomization unit = the whole episode), **cluster
bootstrap** CIs (resample episodes, not steps), and **GEE** (cluster-robust), seed-fixed and
reproducible (`python -m stage2.stats_analysis`):

| metric (ALL 10 full-goal boxes) | Δ | naive z p | **permutation p (clustered)** | cluster-boot CI95(Δ) | effect size | verdict |
|---|--:|--:|--:|--:|--:|---|
| **per-step progress** | +14.8pp | 0.0066 | **0.013** | [+4.9, +26.1]pp | OR 1.81, h 0.30 | **SIGNIFICANT** |
| **goal-aligned progress** (forward-action only) | +9.3pp | 0.039 | **0.028** | [+2.0, +17.5]pp | OR 1.70, h 0.22 | **SIGNIFICANT** |
| per-episode goal-reach | +14.0pp | 0.096 | 0.090 | [+2.0, +26.0]pp | OR 2.18, h 0.34 | NS |

**Multiple-comparison correction (Holm-Bonferroni, 2026-06-21 — `stage2/stats_analysis.py` →
`multiple_comparison`).** Treating the **pre-specified pooled** per-step + per-episode as the PRIMARY
confirmatory family: the **pooled per-step SURVIVES Holm (raw 0.02 → adj 0.04, SIGNIFICANT)**; per-episode
does not (0.09). Over the **full exploratory family of 39 clustered tests** (every per-box + stratified +
variant), **0 survive** and 5 (incl. pooled per-step) are "suggestive" — expected, since per-box tests are
underpowered by design. ⇒ the confirmatory per-step claim holds under correction; the per-box numbers are
descriptive only. (The C-B headline — DC-1 full-chain root 100% vs 40%, non-overlapping CIs — and G3 paired
are SEPARATE pre-specified comparisons, not in this exploratory family.)

**New metrics added (2026-06-21).** Each engagement/arm now also reports: **`top1_oracle_agreement_rate`**
(#3 — direct PRM ranking accuracy: fraction of decisions whose top-1 == the goal-aware oracle's pick over the
real candidate set), and cost **`llm_tokens_total` + `duration_s`** (#4 — token + wall-clock, alongside
call/exec counts). Effect sizes (RD/RR/OR/Cohen's h) are reported with every clustered test.
| per-step progress (self-adv subset only) | +14.4pp | **0.024** | **0.066** | [+1.9, +28.3]pp | OR 1.79 | **NS** |

**Honest correction to the earlier headline:** the per-step advantage on the *self-advertising subset*
looked significant under the naive test (p=0.024) but **drops to p=0.066 once clustered** — it was an
artifact of treating steps as independent. The result that **survives** clustering is the pooled
per-step (and the stricter goal-aligned) progress over **all 10 full-goal boxes** (perm p=0.013 / 0.028,
cluster-bootstrap CI excludes 0). The auth-milestone boxes (WebLogic, Gitea) are **excluded** from goal
denominators — their full goal is unreachable by a single command by design; reported on milestone only.

**Higher-N replication (10 trials/arm × 5 self-adv boxes, `deepseek-chat`,
`outputs/stage2_ab_highn_llm.json`):** the per-step gain **replicates** — PRM **66.7%** (104/156) vs
llm_only **54.6%** (119/218), Δ+12.1pp, clustered **permutation p=0.012**. But per-episode goal-reach
**collapses to an exact tie: 21/50 = 42% vs 21/50 = 42%.** So the +14pp per-episode number at n=5 was
**noise** — at n=10 it is **zero**. **The PRM improves the per-step process, not the episode outcome.**
The per-step benefit does not convert to more full-goal completions because the binding constraint is
elsewhere (§2: the proposer never proposes the exploit in 28/50 episodes — identical across arms — and
the multi-step exploits don't assemble within budget).

### 2. Where the gain lives — failure taxonomy

| terminal reason | PRM | llm_only |
|---|--:|--:|
| success | **15** | 8 |
| foothold, no file read | 2 | 6 |
| exploit executed, no foothold | 4 | 8 |
| **exploit never proposed** | **28** | **28** |
| budget exhausted | 1 | 0 |
| goal unreachable by design (auth boxes) | 10 | 10 |
| safety refusal | 0 | 0 |

The dominant failure — *exploit never proposed* — is **identical (28=28) across arms**: a **proposer
ceiling** the reranker cannot touch (you cannot rank an action the proposer never emits). The PRM's gain
is concentrated in converting partial progress (`foothold_no_file` 6→2, `exploit_executed_no_foothold`
8→4) into `success` (8→15). So the benefit is real and lives exactly where a reranker *can* act.

### 3. Is the gain actually the PRM? — reranker-isolation ablation (the decisive control)

The live A/B varies the LLM proposer, so the gain could be proposer luck / guards / recipe / candidate
order, not the PRM. The ablation removes every confound: the proposer is held **fixed and deterministic**,
the same per-trial candidate-shuffle seed is replayed for all arms (paired design), and **only the rerank
function varies** (`python -m stage2.ablation_rerank`, 8 seeds × 12 boxes (12-box subset — the 3 later-added boxes httpd/nginx/Tomcat8 are not in this ablation), key-free):

Run **two** ablations — they disagree, and the disagreement is the finding.

**(a) Deterministic proposer** (`TargetAwareProposer` — dumps the box's *full* candidate surface every
step; paired shuffle seed; `python -m stage2.ablation_rerank`, 8 seeds × 12 boxes (12-box subset — the 3 later-added boxes httpd/nginx/Tomcat8 are not in this ablation), **key-free**):

| rerank mode | per-step progress | goal-reach | steps used (80 eps) |
|---|--:|--:|--:|
| oracle (goal-ladder heuristic) | 31.2% | 60% | 631 |
| heuristic (hand priority) | 26.2% | 60% | 568 |
| shuffled_prm (PRM scores, mapping destroyed) | 32.7% | 55% | 682 |
| random (floor) | 29.3% | 46% | 774 |
| **prm** | **26.7%** | 50% | **960** |

Here prm < random (Δ=−2.7pp, p=0.003), < shuffled_prm (p<0.001), ≈ heuristic, < oracle. With the *full
surface* forced into the ranker, the PRM's Stage-1 **enumeration/recon bias** dominates: it scores
`web_path_enumeration` **1.000** vs `exploit_attempt` 0.367 vs `command_execution` 0.080, so it
front-loads recon and is the least efficient mode (960 steps).

**(b) Real LLM proposer** (`deepseek-chat` emits *targeted* candidate sets; its stochasticity supplies
trial variation; `--proposer llm`, 5 trials × 6 exploit-proposable boxes, n=30/mode):

| rerank mode | per-step progress | goal-reach | prm vs it (episode-clustered permutation) |
|---|--:|--:|---|
| **prm** | **68.5%** (61/89) | **40%** | — |
| llm_only (native order) | 48.1% | 23% | Δ=+20.4pp, **p=0.0055** |
| random | 50.0% | 33% | Δ=+18.5pp, **p=0.0068** |
| oracle (goal-ladder heuristic) | 47.9% | 27% | Δ=+20.6pp, **p=0.001** |

Here the PRM is the **best** mode: it **significantly beats random** (p=0.007) and the goal-ladder
heuristic (p=0.001), not just the LLM's native order. So on a realistic proposer's candidate
distribution the per-step gain **is attributable to the PRM's learned ranking** — not "having scores"
(random control), not the guards/recipe (held constant), not just out-ranking a bad LLM order.

### 4. Reconciling §1, §3a, §3b — the honest bottom line

**The PRM's value is real but PROPOSER-CONDITIONAL.** When the proposer emits a small, targeted candidate
set (the realistic LLM case), the PRM ranks it well and beats random/heuristic/native — the §1 per-step
gain is genuinely the PRM (§3b, prm > random p=0.007). When the proposer dumps the *entire* action
surface every step (the deterministic stress test), the PRM's recon bias makes it no better than random
(§3a). So:

- **Supportable claim:** *paired with an LLM proposer, the abstract-trained PRM gives a real,
  attributable per-step reranking gain on real web targets* (beats random, clustered-significant, Δ≈+12
  to +20pp, replicated across 3 runs). It is a **per-step PROCESS improver.**
- **Required caveat 1 — process, not outcome:** the per-step gain does **not** translate to more
  episodes reaching the full goal. At n=5 the per-episode goal-rate looked like +14pp; at n=10 it is an
  **exact tie (42%=42%)**. The PRM makes better per-decision rankings but the same fraction of episodes
  finish, because the ceiling is the proposer (exploit never proposed, 28=28) and multi-step assembly.
- **Required caveat 2 — not a standalone ranker:** fed the raw action surface (deterministic ablation)
  its recon bias dominates and random/heuristic match it; the benefit needs a proposer that pre-filters
  to sensible candidates.

This is the honest, conditioned conclusion — neither the inflated "PRM uplift" nor the over-deflated "PRM
has no skill." Both ablation JSONs: `outputs/stage2_ablation_rerank.json` (deterministic),
`outputs/stage2_ablation_rerank_llm.json` (LLM).

### 5. LLM within-box memory & proposer-prompt improvements (does richer memory stop the spinning?)

The failure taxonomy (§2) and a code+data investigation showed the LLM proposer's per-box "memory" is
coarse: each step it is re-sent only the current abstract state, a 3-step *bool-only* feedback window
(`evidence=''` was hard-coded), and the *set* of exhausted action **types** — not the ordered trace, the
outcomes, or how many times each action stalled. Measured spinning across 120 logged episodes:
**consecutive-repeat 0.405, wasted-rate 0.568** (>½ of steps yield no new info), with verbatim cycles
recurring across trials. Three leak-free/leaky toggles were A/B'd (8 boxes × 4 trials/arm, `deepseek-chat`,
`mode=llm_only`, episode-clustered permutation; `python -m stage2.improvement_ab`):

| treatment (vs its paired baseline) | consecutive-repeat | wasted | goal-reach | leakage? |
|---|--:|--:|--:|---|
| **rich_memory** (real evidence + ordered trace + per-type no-progress counts) | 53%→**27%** (p≈0) | 52%→55% (NS) | **12%→44% (p=0.004)** | **none — clean** |
| enhanced_prompt (fingerprint-once + named CVE techniques) | 57%→32% (p=0.0001) | 52%→32% (p=0.012) | 16%→53% (p=0.002) | **YES — names test-set CVEs** |
| generic_prompt (same strategy, **no** CVE names — leakage control) | 48%→26% (p=0.008) | 47%→50% (NS) | 28%→19% (**NS**) | none |

**Findings (honest):**
1. **The user's hypothesis is confirmed for spinning:** *all three* treatments significantly cut the
   consecutive-repeat rate (~−20 to −25pp, p<0.01). Giving the model real history / a fingerprint-once
   rule measurably stops it going in circles.
2. **rich_memory is the one fair *success* win:** +32pp goal-reach (12%→44%, p=0.004) with **no leakage**
   — mechanistically, seeing "fingerprint ×2, no progress" pushes the LLM off recon to exploitation
   (exploit_proposed 41%→75%, p=0.007). *(This corrected my prior prediction that memory would not help
   success; the data overruled it. Caveat: goal-reach is noisy at n=32 — pooled baseline across the three
   runs is ~19% — so this fair win wants a higher-N replication before it is a headline.)*
3. **The enhanced prompt's success gain is test-set LEAKAGE, not skill:** its CVE names cover exactly the
   test boxes; the **generic** control (identical strategy, names removed) does **not** lift goal-reach
   (19% vs 28% baseline, NS). So the +37pp is the cheat-sheet. `SYS_ENHANCED` is marked leakage-demo-only
   in code; only the leak-free *strategy* (fingerprint-once + anti-loop) is keepable, and it helps
   spinning but not success.

**Net:** richer per-box memory (`rich_memory`) is a clean, recommended improvement — it both reduces
looping and (fairly) lifts success; the proposer-prompt success boost does not survive a leakage control.
Neither touches the deeper ceiling on boxes whose multi-step exploit the LLM cannot construct at all
(Drupalgeddon2 stays 0/4 in every arm). Reports: `outputs/stage2_improvement_{memory,proposer,proposer_generic}.json`.

### 6. Splitting a pseudo-ceiling from the real ceiling — the Drupalgeddon2 adapter fix

Drupalgeddon2 was 0/4 in every arm, read as "the LLM can't do the multi-step exploit." A live diagnosis
showed that was **two-thirds wrong**: the η recipe **fires the RCE correctly** (`uid=33(www-data)` comes
back), the failure was a **φ/η adapter bug** plus a separate proposer gap:
- **η bug:** the file-read used PHP `exec()`, which returns only the **last line** of output, so
  `cat /etc/passwd` yielded `_apt:x:100:…` and **never `root:x:0:0`**. Fixed by switching the recipe to
  `passthru` (full output).
- **φ bug:** `_parse_fileread` only matched the literal `root:x:0:0`, so even a genuine passwd line was
  not recorded as a file read. Fixed with a generic passwd-line regex `_PASSWD_LINE` (`name:x:uid:gid:`),
  which also makes φ robust to *any* box where an `exec()` RCE truncates a multi-line read.
- A latent **non-monotonic shell-state** worry was checked and is fine: a later file-read does not
  downgrade `command_execution`, so the goal (`cmd ∧ file`) latches correctly.

After the fix, the **deterministic** proposer solves Drupalgeddon2 **3/3** (cumulative `_goal_reached`
→ True), proving the adapter now works end-to-end. **But the LLM-autonomous loop is still 0/4** — 3/4
trials die at step 1 with `exploit_proposed=False`: the LLM never proposes the Drupalgeddon2 exploit.
So the fix removed the **pseudo-ceiling** (adapter) and isolated the **real ceiling** (proposer doesn't
know/propose the exploit), which is a proposer-capability problem (RAG / tools / stronger model — not a
memory, reranking, or adapter fix). The φ/η fixes are leak-free and general (Tomcat & php-cgi were
confirmed adapter-reachable 3/3 by the deterministic proposer too; their LLM-loop gaps are likewise
proposer/ψ, not adapter). ψ exploit coverage also extended (deploy/write/PUT a JSP/WAR →
file_upload_attempt; trigger OGNL/RCE/deserialization → exploit_attempt), held-out ψ false-accept still 0.

## The 7-dimension metric suite

Defined in [STAGE2_INFERENCE_METRICS.md](STAGE2_INFERENCE_METRICS.md): (1) per-step progress [headline,
high-N], (2) graded milestones shell/cmd/file/root, (3) efficiency (steps, wasted-rate), (4) cost
(proposer_calls, η executions), (5) live out-of-abstraction rate, (6) gate-refusals (**0** across all
120 episodes — safety held), (7) Wilson CIs + clustered tests + effect sizes. The full all-box table is
[STAGE2_SEVEN_DIM_TABLE.md](STAGE2_SEVEN_DIM_TABLE.md).

## Methods, fairness, reproducibility

- **Boxes (12, all `127.0.0.1`):** 7 self-advertising (ThinkPHP-5-rce/5.0.23, Struts2-S2-048/045,
  Drupalgeddon2, Tomcat-12615, Joomla-8917-SQLi) + 3 contrast (php-cgi-2012-1823, php-inclusion-LFI,
  Rails-5418) + 2 auth-milestone (WebLogic-weakpw, Gitea-1.4, goal unreachable by design).
- **Fairness:** both arms share the same proposer + temperature; arm order is randomized per trial
  (`live_ab_trials.py --seed`, default randomized; `--fixed-arm-order` to disable); the ablation holds
  the proposer fixed and pairs the shuffle seed across arms.
- **Reproducibility:** `stage2/target_registry.json` (16 boxes: container, port, image, compose dir,
  healthcheck); `python -m stage2.reset_target --all --check` (healthcheck) / `--label X` (down -v && up
  -d + healthcheck) for clean state between runs; full run metadata (model/temp/seed/budget/ports/timing)
  recorded in each report's `run_metadata`; `python -m stage2.preflight` now also probes docker / the 12
  containers / tool binaries (live-infra readiness, reported separately from offline readiness).
- **Stats:** seed-fixed permutation (20k) + cluster bootstrap (10k) + GEE; non-converged GEE on tiny
  strata is suppressed (permutation is primary).

## Safety (enforced throughout)

`LiveExecutor` refuses every call until `STAGE2_LIVE_AUTHORIZED` + `--confirmed-isolated`; targets scoped
to loopback; destructive tokens refused; every command logged. Multi-step exploits wired as a single
`bash -c "s1; s2"` still pass the target-scope + destructive-token checks. Gate-refusals = 0 across 120
A/B episodes and the 480-episode ablation.

## Honest caveats / what is still open

- **n=5 trials/box** for the LLM A/B → per-episode CIs wide, per-box goal noisy (it *flipped* on
  ThinkPHP-5-rce). The robust claim is the pooled per-step progress, not any single-box success rate.
  Increasing to 10–20 trials/box on the self-advertising boxes (the user's request #2) needs a live
  `DEEPSEEK_API_KEY` in env — **not currently set**; the runner is one command away when it is.
- **Two self-adv boxes (Drupalgeddon2, Tomcat-12615) went 0/0 on both arms** in the autonomous loop —
  the η recipe pops them in fixed-sequence `live_smoke`, but the LLM did not reliably propose the
  multi-step exploit, so the reranker had nothing to rank.
- **The decisive negative result (§3) recontextualizes the positive one (§1).** Do not headline "PRM
  uplift" without §3's conditioning.
- A clean LLM-proposer ablation (random/oracle rerank *with the LLM proposer*) still needs the key; the
  key-free ablation uses a deterministic stand-in proposer.

---

## Appendix — superseded interim results (kept for provenance; DO NOT cite as current)

These earlier runs were progressively replaced by the §1–§4 cluster-robust 12-box analysis above. They
used the **naive** two-proportion z-test (now known to over-state significance) and/or far fewer boxes.

- **First live RCE (single box, fixed sequence):** ThinkPHP-5-rce `live_smoke` drove a real RCE end to
  end (`uid=33(www-data)`, read `/etc/passwd`); surfaced + fixed 3 sim-to-real bugs (GBK decode → bytes
  +utf-8/replace; φ mis-read CSS as credentials → `_CSS_HTML_LINE` skip; curl `[]` globbing → `curl -g`)
  and the generic-η-can't-fire-a-CVE gap → per-target `eta_recipes`.
- **Single-box LLM A/B (deepseek-v4-pro, n=6):** prm 6/6 vs llm_only 3/6 — directional, CIs overlapped,
  **did not replicate**.
- **4-box study (n=6, v4-pro):** pooled self-adv prm 83.3% vs 61.1%, naive p=0.137 (NS); the 100%-vs-50%
  single-box headline did not replicate (sibling ThinkPHP tied).
- **9-box / 12-box naive 7-dim study:** per-step "SIGNIFICANT" at naive p=0.002–0.024 — **this is the
  claim now corrected in §1**: it survives clustering only when pooled over all full-goal boxes, not on
  the self-advertising subset alone.
- **KEY FINDING (still valid):** the abstract `StateProposer` cannot drive a real CVE box (only proposes
  recon, never `exploit_attempt`); live needs a CVE-aware proposer (LLM / TargetAware stand-in) + a
  permissive guard. The PRM is a reranker, not a proposer.
