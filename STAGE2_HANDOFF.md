# STAGE 2 HANDOFF — Real-VulnHub Inference Stage

**Audience:** the next operator (human or AI). Read this top-to-bottom and you can resume the
experiments immediately. Last updated 2026-06-18. Working dir for everything below:
`E:\PT\PT+LLM+检测\单主机web服务\WebAttackSim` (call it `$ROOT`).

> **Prime directive of this project: honesty over headline metrics.** Every number here has been
> deflated on purpose where the statistics or controls did not hold. Do NOT inflate. When you run new
> experiments, report negative results, leakage, and noise as plainly as the wins. The history below is
> a sequence of "we found a nice number → we attacked it with a control → we kept only what survived."

---

## 0. TL;DR — the one-paragraph state

A two-stage system. **Stage 1** (frozen) trained, in an abstract single-host-web simulator, a DQN value
oracle → a **Pentest-PRM** (a per-step *reranker*, NOT a policy). **Stage 2** (current work) runs an
autonomous pentest loop against **12 real Vulhub Docker boxes** on `127.0.0.1` via a φ/ψ/η adapter, and
A/B-tests whether the PRM reranking the proposer's candidates beats the proposer's own order. **Honest
bottom line so far:** with a real LLM proposer the PRM gives a *real, attributable, clustered-significant*
**per-step process** improvement (beats random reranking, p≈0.007), but it does **not** improve
**episode success rate** (tied at higher N) and is **not** a standalone ranker (fed the raw action surface
its recon bias makes it ≤ random). Separately, a `rich_memory` upgrade to the LLM proposer's per-box
memory **fairly** lifts both spinning and success; a CVE-named prompt also lifts success but that was
**test-set leakage** (busted by a generic-prompt control). The dominant remaining failure is
`exploit_never_proposed` (the proposer never puts the exploit on the table) — a proposer-capability
ceiling that memory/reranking cannot touch.

---

## 1. What this project is

- **Goal:** evaluate whether an abstract-trained process-reward model (PRM) + value prior transfers to
  steering a real web-pentest agent, *honestly* and *without leaking* the answer the agent should discover.
- **Stage 1 (done, frozen — see `TRAINING_STAGE_SUMMARY.md`, `REPORT_STAGE1_RL_PRM_robustness.md`):**
  abstract 16-action single-host-web MDP, programmatic task generator (12 chain-topology families),
  permissive DQN value oracle, gated label generation, `prm_strong.joblib` (a state-conditioned PRM),
  Q* verification, leakage audit, calibration. **Do not retrain Stage 1** unless explicitly asked — the
  artifacts must stay reproducible. The 16-action schema is FROZEN.
- **Stage 2 (current — `STAGE2_PLAN.md`, `STAGE2_LIVE_RESULTS.md`):** the φ/ψ/η adapter + safety gate +
  the autonomous engagement loop + all the live A/B and ablation experiments. **This is where new work
  happens.**

**Method constraints (hard rules):**
- RL/PRM is trained ONLY in the abstract simulator. Real boxes are *adapter-validation* targets, not RL
  training sources.
- The simulator must not leak hidden task ground truth into PRM input.
- PRM input = observable context + action text + normalized action + feedback history. **Never** oracle
  q-values as features.
- **No leakage in Stage 2 either:** do not inject "box X is vulnerable via Y" into φ or the proposer
  prompt. The agent must discover/propose the exploit. (We learned this the hard way — see §5.7.)

---

## 2. Repo & environment

- **OS:** Windows 11. Two shells available: **PowerShell** (primary) and **Git Bash** (`bash` tool).
  They have different syntax — see §10 gotchas.
- **Python:** Anaconda at `D:\software\anaconda3`. Run scripts as modules from `$ROOT`
  (`python -m stage2.<name>`). Libs present: numpy, scipy, statsmodels, sklearn, pandas, joblib.
- **Tests:** `python -m pytest tests\ -m "not slow" -q` → **393 passed, 1 skipped** (keep it green).
- **DeepSeek API key:** the LLM proposer needs `DEEPSEEK_API_KEY` in the **environment only**. NEVER
  write it to a file or print it. The reader (`scripts/deepseek_client.py`) reads it from env. Working
  models: `deepseek-chat` (fast, default for all studies — both arms share it so it's fair),
  `deepseek-reasoner`, `deepseek-v4-pro` (slow, ~6h/box — avoid), `deepseek-v4-flash`. NOT `deepseek-v4`.
  The user supplies the key per-session and is told to rotate it afterward.
- **Live-execution authorization (safety gate):** every live run requires BOTH
  `STAGE2_LIVE_AUTHORIZED=i-own-this-isolated-authorized-lab` in env AND the `--confirmed-isolated` CLI
  flag. Without them the executor refuses. Kill switch: set `STAGE2_KILL_SWITCH` to abort mid-run.
- **Containers:** 12 Vulhub web boxes on `127.0.0.1:8080-8091` (+ a few helper containers). They are
  long-running. Check health (non-destructive): `python -m stage2.reset_target --all --check`. Restart
  one clean: `python -m stage2.reset_target --label <Label>`. Registry of all 12 (container, port,
  image, compose dir, healthcheck): `stage2/target_registry.json`. Compose dirs live under
  `E:\PT\vulhub\...`. Docker Hub direct pulls FAIL (TLS/EOF) — use the daocloud mirror
  `docker.m.daocloud.io/...` + a resume-retry loop, then `docker tag` to the `vulhub/*` name (§10).

Example env setup for one live command (PowerShell):
```powershell
$env:STAGE2_LIVE_AUTHORIZED="i-own-this-isolated-authorized-lab"
$env:DEEPSEEK_API_KEY="<key>"        # only if the proposer is the LLM
python -m stage2.live_ab_trials --target stage2\targets\thinkphp-5-rce.json --proposer llm `
    --model deepseek-chat --trials 5 --budget 12 --confirmed-isolated
```

---

## 3. Architecture — the φ/ψ/η loop

The autonomous engagement (`stage2/engagement.py::run_engagement`) is a budgeted loop:

```
propose → ψ-normalize → (PRM rerank | proposer order) → η-render → gated LiveExecutor → φ-observe → repeat
```

- **Proposer** — produces candidate next actions as natural-language strings. Three implementations:
  - `LLMProposer` (DeepSeek, gated, the real one). Stateless per call: each step sends a fresh prompt.
  - `TargetAwareProposer` (offline, deterministic) — emits the box's declared candidate surface from the
    descriptor. A key-free stand-in used for *reranker-isolation* ablations.
  - `StateProposer` (offline) — abstract-MDP candidates. **Cannot drive a real CVE box** (only proposes
    recon, never `exploit_attempt`). Don't use it for live.
  - `CachingProposer(inner)` — a wrapper that memoises `propose(context, obs)` by the exact decision
    state (common random numbers). Share ONE instance across rerank arms and they reuse the SAME candidate
    set wherever their states coincide (always step 0 + the locked prefix), so the only difference is the
    ranking and the LLM is called once per unique state (~half the calls). This is the **paired A/B**
    design (`stage2/paired_ab.py`); exact pairing holds on shared states only — divergent steps query
    fresh (CRN, not total pairing). Exposes `.hits/.misses/.cache_stats()`.
- **ψ (normalizer)** — text → one of the 16 abstract actions. Two layers: the frozen Stage-1
  `normalize_llm_action`, wrapped by `stage2/psi.py::EnhancedNormalizer` (a Stage-2-local recovery layer
  that only acts on Stage-1 `unsupported`, with an out-guard that keeps non-web primitives out; held-out
  accuracy 49%→78.5%, false-accept 0). **Important coupling:** the frozen PRM's candidate features use
  the *training-time Stage-1 ψ* (`_prm_scores`), while the *executed/mapped* action uses the enhanced ψ.
  Do not swap the PRM's normalizer to the enhanced one without retraining the PRM (it goes OOD →
  rerank top-1 0.5→0.1).
- **η (renderer)** — abstract action → concrete shell command, per-target via the descriptor's
  `eta_recipes` (`stage2/eta.py`). Brace-robust (`str.replace`, not `.format`, so OGNL/JSP `{}` payloads
  wire). Multi-step exploits are wired as a single `bash -c "step1; step2"`. The recipe HOLDS the
  box-specific payload — the LLM only needs to propose the abstract action.
- **LiveExecutor** (gated, `stage2/eta.py`) — runs the command via subprocess, captures bytes (decodes
  utf-8/replace to survive Windows GBK), audits every call to `outputs/stage2_*_audit.jsonl`. Refuses
  until authorized (§2). Target scoped to loopback; destructive tokens denied (`stage2/safety.py`).
- **φ (observer, `stage2/phi.py`)** — parses real tool output back into the AbstractWebState (services,
  tech_stack, paths, creds, shell_state, read_files, privilege_level). This is the messy-real-output
  layer; most live bugs live here.
- **PRM rerank** — `_prm_scores(prm, context, candidates)` scores candidates; mode `prm` picks argmax,
  mode `llm_only` keeps proposer order. Other modes for ablation: `random`, `shuffled_prm`, `heuristic`,
  `oracle` (see `_choose_index`).
- **Goal** (`_goal_reached`): `privilege_level==root` OR a root/flag file read OR
  (`shell_state==command_execution` AND `read_files` non-empty). Checked at loop top AND after each
  action (so a goal assembled on the final step is credited).
- **Termination / stopping (the loop CANNOT run forever).** `run_engagement` ends with one of these
  `stop_reason`s:
  - `budget_exhausted` — hard cap `for t in range(budget)` (default 12). Ultimate backstop.
  - `goal_reached` — checked at loop top + after each action.
  - `kill_switch` — `STAGE2_KILL_SWITCH` env set.
  - `no_available_action` — every proposed candidate is unmappable or already exhausted → empty `avail`.
    (In the recorded 120-episode study this was 95/120 of terminations — the de-facto stuck-stop.)
  - `no_progress_stuck` — the **global no-progress circuit-breaker** (added 2026-06-18): `patience`
    (default **4**, CLI `--patience`, `0` disables) consecutive steps with NO new information across
    **any** action type → the agent is judged stuck and gives up early. This catches **cross-type
    oscillation** (A→B→A→B) that the per-type exhaustion below misses, and produces a clean, interpretable
    "I'm spinning, stop" label. Resets to 0 on any productive step; the goal check takes precedence.
- **Anti-loop machinery (two layers):** (1) per-TYPE — an action type that yields no new info
  `NO_PROGRESS_LIMIT=2` times in a row is added to `exhausted` and masked out (and told to the proposer
  via `_tried`); (2) GLOBAL — the `patience` circuit-breaker above, across all types. The per-type guard
  rarely fires the *termination* itself (in the study `no_progress_exhausted`≈0); it masks, and the empty
  `avail` (no_available_action) or the global breaker (no_progress_stuck) ends the episode.

### Efficiency improvements — Tier 1 (2026-06-21, default ON)
Detail-level changes (no architecture change) that cut wasted steps + LLM cost:
1. **φ content-credit pass** (`phi.py::_content_credit`, called in `ingest`): tool-agnostic, monotone
   credit — any output containing `uid=…(name)` → command_execution (+root if uid=0), a `/etc/passwd`
   line → read_files, a `flag{…}` → read_files. So a result returned via the "wrong" action (file via an
   exploit action, etc.) is credited immediately, killing a large class of false no-progress steps.
   *Measured (deterministic): drupal 5→2 steps, struts2-048 1 step→root, 0% wasted on goal-reaching boxes.*
2. **`rich_memory` default True** in `run_engagement` (was a measured win: repeats 53%→27%, no leakage).
3. **CRN paired A/B default** in `live_ab_trials` (`CachingProposer` shared per trial; `--unpaired` to
   disable): arms reuse the candidate set on coincident states → ~half the LLM calls + lower variance;
   `mean_cache_hit_rate` recorded in `run_metadata`.
**Tier 2 (2026-06-21) — split outcome, kept only what an A/B proved:**
- ✅ **KEPT: action-vocabulary hint** in `LLMProposer._enrich` (lists the 16 action types so the LLM
  proposes mappable steps). Measured win: live out-of-abstraction rate **40–64% → 0–8%** → far fewer
  unmappable proposals, no extra LLM calls, no leakage (it's the agent's own action space).
- ❌ **REJECTED: phase-aware PRM reweight** (`_phase_weight`, downweight recon once fingerprinted). An A/B
  showed it **regressed the PRM arm** (thinkphp per-step 80%→50%; struts2-048 prm 40%→25% vs llm_only
  100%) — the hand-set weights mis-fire when the down-weighted action is the goal step. Reverted;
  `_phase_weight` is kept in code with a REJECTED note, not used. A gentler/learned prior is future work.
- ❌ **DROPPED as redundant: `_fields_gained`-driven exhaustion** — after the content-credit fix the state
  is monotone, so `made_progress ≈ fields_gained>0`; the change would be a no-op.

**Tier 3 (2026-06-21):**
- ✅ **KEPT: milestone-slack patience** (`run_engagement`): reaching a NEW milestone (shell→cmd→file→root)
  grants a bounded reprieve (`MILESTONE_SLACK=3`) to the no-progress counter, so a genuine multi-step chain
  is not truncated while pure spinning still trips `patience`. Verified: **httpd now reaches goal at the
  default patience=4** (was `no_progress_stuck`); no regression on simple boxes (struts2 1 step, thinkphp 2
  steps) — it only ADDS slack, never changes which action is chosen (so it cannot regress like the rejected
  phase-reweight). Note: Tier-1's content-credit had already fixed tomcat8's truncation.
- **NOT applied: precondition-aware action ordering** — it is the same class of hand-set action prior that
  the rejected phase-reweight (Tier-2 A) showed backfires; not added without an A/B win.

**★ CONFIRMED finding — the efficiency work INVERTS the PRM result (6-box A/B, `outputs/eff_*.json`):**
with the full efficiency stack (esp. the vocab hint making the LLM's *native* ordering good), **`llm_only`
significantly BEATS `prm` on efficiency**: per-step **66.7% vs 39.6%, Δ=−27.1pp, clustered permutation
p<0.0001**; wasted **33% vs 60%**; goal-reach **TIED at 93%** (28/30 each). The PRM reranks the now-good LLM
order toward recon (its bias), wasting ~2× the steps for the same success. Compare the *original* (weak
proposer) result: PRM > llm_only per-step at p=0.013. So **improving the proposer obsoletes — and inverts —
the reranker.** This is the honest end-state of the efficiency work and it sharpens the long-standing "PRM
is proposer-conditional, no intrinsic skill" conclusion. **Efficiency recommendation: run `--mode llm_only`
with the Tier-1/2/3 stack (content-credit + rich_memory + CRN + vocab hint + milestone-slack); do NOT use
the PRM rerank when the proposer is vocab-hinted.** (The PRM/RL Stage-1 remains valuable as the abstract
artifact and as the reranker for a *weak* proposer — its value just doesn't survive a good proposer.)

**Can the PRM be fixed to beat llm_only? THREE attempts, ALL FAILED (2026-06-21) — the recon bias is robust.**
Root cause: the PRM's training labels over-value recon (`web_path_enumeration` mean-label **0.887** vs
`exploit_attempt` 0.535) because the *masked* abstract oracle rarely produced "recon-when-already-known"
states, so the PRM extrapolates recon's high value to real targets where it is wasteful.
- ❌ **Inference reweight** (Tier-2 A): regressed.
- ❌ **Label-correction retrain** (`scripts/retrain_prm_efficiency.py` → `prm_strong_v2.joblib`): augmented
  the data with "recon-when-advanced → low value" negatives (3642, 3× weight) + retrained the score model.
  Bias **did not shift** (advanced-state web_path_enum stayed **0.98** ≫ exploit **0.45**) — the
  gradient-boosted model + coarse count features won't unlearn it from labels alone.
- ❌ **Top-K shortlist restriction** (rerank only the proposer's top-3): made it **WORSE** (prm 0% goal on
  thinkphp-5023 / struts2-048) — a recon action sits inside the top-3, so the PRM still picks it.
**Conclusion:** the recon bias is not removable by surgical (inference- or label-level) methods. A genuine
fix needs a fundamental retrain — features that explicitly encode recon-redundancy and/or an oracle reward
with a step/efficiency cost (regenerate Stage-1 labels + retrain oracle+PRM; large, uncertain). The honest
working recommendation stands: **`--mode llm_only`**. `prm` mode is plain argmax; `prm_strong_v2.joblib` +
the retrain script are kept for the record (do not adopt without an A/B win).

**Per-box heterogeneity (don't over-flatten):** the PRM is a HIGH-VARIANCE reranker — it **helps** when the
proposer's order is bad and **hurts** when it's good. Top-K 6-box A/B goals (prm vs llm_only): thinkphp
40/100, tp5023 0/100, struts2-048 0/100, struts2-045 100/100, drupal 100/100, **joomla 60/0 (PRM RESCUED a
box where llm_only failed)**. So it's net-negative *pooled* (most boxes now have a good LLM order thanks to
the vocab hint), but the PRM still salvages the occasional bad-order box. This is the proposer-conditional
thesis at the per-box level — a router that used PRM only when the proposer is uncertain could in principle
capture both, but detecting "is this order good?" reliably is itself unsolved.

---

## 4. The 15 live targets (12 original + 3 added 2026-06-21)

Ports/containers in `stage2/target_registry.json`. `recipes work?` = the deterministic `TargetAware`
proposer reaches goal (i.e. the η/φ adapter is sound). `self-adv` = the stack fingerprints itself.

| label | port | class | self-adv | goal reachable by design | recipes work? | notes |
|---|--:|---|:--:|:--:|:--:|---|
| ThinkPHP-5-rce | 8080 | RCE | Y | Y | yes | `invokefunction` `%5C` payload, `curl -g` |
| ThinkPHP-5.0.23 | 8084 | RCE | Y | Y | yes | captcha `_method=__construct` (INVERSE of 5-rce) |
| Struts2-S2-048 | 8083 | RCE | Y | Y | yes | OGNL `/integration/saveGangster.action` |
| Struts2-S2-045 | 8089 | RCE | Y | Y | yes | Content-Type header OGNL |
| Drupalgeddon2 | 8081 | RCE | Y | Y | yes (fixed) | AJAX form-render; needs `passthru` (see §5.8) |
| Tomcat-12615 | 8087 | RCE(upload) | Y | Y | yes | PUT `.jsp` then GET `?cmd=` |
| Joomla-8917 | 8086 | SQLi | Y | Y | yes | `list[fullordering]=updatexml(...)`, `-g` |
| php-cgi-2012-1823 | 8082 | RCE | **n** | Y | yes | `?-d+auto_prepend_file=php://input`; non-self-adv |
| php-inclusion-LFI | 8085 | LFI | n | Y | yes | `/lfi.php?file=/etc/passwd` |
| Rails-5418 | 8088 | LFI | n | Y | yes | `Accept: ../../../etc/passwd{{` header |
| WebLogic-weakpw | 8090 | auth | n | **N (auth-milestone)** | auth only | weblogic/`Oracle@123`, 2-step `bash -c` |
| Gitea-1.4 | 8091 | auth | n | **N (auth-milestone)** | auth only | CSRF-chained register; RCE multi-step deferred |
| **Tomcat8-weakpw** | 8094 | RCE | Y | Y | **yes (→root)** | manager tomcat/tomcat → deploy `stage2/payloads/tomcat_shell.war` (`/manager/text/deploy`) → `/sh/cmd.jsp?cmd=` = uid=0(root). Real multi-step ②→⑤→⑥ chain. |
| **httpd-41773** | 8095 | RCE | Y | Y | **yes** | CVE-2021-41773: file-read `/icons/.%2e/..//etc/passwd`; RCE POST `/cgi-bin/.%2e/..//bin/sh` (mod_cgi). ④+⑥ |
| **nginx-insecure** | 8097 | LFI | Y | **N (file-read milestone)** | file-read only | off-by-slash `/files../etc/passwd` (alias `/home/`). ①+④ recon→exposure. **No RCE** (nginx) → reaches read_files milestone only. |

**Milestone-only boxes** (excluded from full-goal-rate denominators, reported separately):
- **Auth-milestone** (WebLogic, Gitea): reach only the login milestone (full RCE is multi-step, deferred).
- **File-read-milestone** (nginx-insecure): reads a sensitive file but has NO command execution by design.
  (php-inclusion / Rails are *also* file-read-only but were historically left in the `reachable=True` pool
  where they score 0 — a known minor inconsistency.)

**Honest finding (patience × multi-step):** the `patience=4` circuit-breaker (§3) **truncates genuine
multi-step chains** (Tomcat8 needs ~8 steps, httpd ~9, with several legitimate no-progress steps between
productive ones), so the 3 new boxes are run with **`--budget 16 --patience 6`**. This is a real tension:
the breaker that stops spinning also limits long chains — tune `patience` to the expected chain length.
All 3 were verified end-to-end with the deterministic proposer (patience disabled): httpd goal+cmd+file,
Tomcat8 goal+cmd+**root**+file, nginx file-read only (correct — no RCE).

---

## 5. What's been done & the honest findings

All Stage-2 experiments use 5 trials/arm unless noted, `deepseek-chat`, both arms sharing the proposer.

### 5.1 Adapter built & gated
φ/ψ/η + `safety.py` AuthorizationGate + functional gated LiveExecutor + 12 target descriptors. Offline
Phase-1 replay on 7 hand-authored VulnHub-class fixtures (`stage2/walkthroughs/`) found: the 16-action
schema covers ~92% of real steps (out-of-abstraction only 8.5% → **do NOT extend the schema**); the
bottleneck was ψ (49%→78.5% after the enhanced layer). Reports: `STAGE2_PHASE1_REPORT.md`.

### 5.2 12-box live A/B + 7-dimension metrics
`STAGE2_INFERENCE_METRICS.md` defines the suite: (1) per-step progress [HEADLINE, high-N], (2) graded
milestones shell/cmd/file/root, (3) efficiency (steps, wasted-rate), (4) cost (proposer_calls, η execs),
(5) live out-of-abstraction rate, (6) gate-refusals (0 across all runs), (7) Wilson CIs + tests. Per-box
JSON: `outputs/stage2_ab_trials_*.json` + `outputs/stage2_live_ab_trials.json`. All-box table:
`STAGE2_SEVEN_DIM_TABLE.md`. Pooled per-step PRM 48.9% vs llm_only 38.6%.

### 5.3 ★ Cluster-robust statistics (the most important methodological fix)
`stage2/stats_analysis.py` → `outputs/stage2_stats_analysis.json`. A naive two-proportion z-test treats
every step as independent — it is not (steps within an episode/box are correlated), so it OVER-states
significance. Re-tested with an **episode-clustered stratified permutation test** + **cluster-bootstrap
CIs** + **GEE**, seed-fixed. Result:
- per-step progress, ALL 10 full-goal boxes: Δ+14.8pp, **perm p=0.013**, boot CI [+4.9,+26.1]pp → **SIG**.
- goal-aligned (forward-action only) progress: Δ+9.3pp, perm p=0.028 → **SIG**.
- self-adv subset per-step: naive p=0.024 → **clustered p=0.066 → NOT sig** (the naive headline was inflated).
- per-episode goal-reach: NOT significant either way.
Failure taxonomy: dominant bucket `exploit_never_proposed = 28 vs 28` (IDENTICAL across arms — proposer
ceiling). PRM's gain = converting `foothold_no_file` (6→2) and `exploit_executed_no_foothold` (8→4) into
`success` (8→15).

### 5.4 Reranker-isolation ablation — deterministic proposer (`outputs/stage2_ablation_rerank.json`)
Hold a deterministic proposer fixed, vary ONLY the rerank function (paired shuffle seeds, key-free). PRM
per-step 26.7% is WORSE than random 29.3% (perm p=0.003), shuffled_prm 32.7% (p<0.001), ≈ heuristic, <
oracle. Diagnosis: PRM scores `web_path_enumeration`=1.000 ≫ `command_execution`=0.080 — the Stage-1
**recon bias**, confirmed live. So fed the *full action surface*, the PRM has no edge.

### 5.5 Reranker-isolation ablation — REAL LLM proposer (`outputs/stage2_ablation_rerank_llm.json`)
With the LLM's *targeted* candidate sets (6 exploit-proposable boxes), the PRM is the BEST mode: per-step
**68.5% vs random 50.0% (perm p=0.0068)**, vs llm_only 48.1% (p=0.0055), vs oracle 47.9% (p=0.001). So
the gain IS attributable to the PRM **on a realistic proposer's candidates** — not "having scores"
(random control), not guards/recipe. **Reconciliation (§5.4 vs §5.5): the PRM's value is real but
PROPOSER-CONDITIONAL** — it helps rank a small targeted set, not the raw surface.

### 5.6 High-N replication (`outputs/stage2_ab_highn_llm.json`)
10 trials × 5 self-adv boxes. per-step **replicates**: PRM 66.7% vs llm_only 54.6%, perm p=0.012. BUT
per-episode goal-reach **collapses to an exact tie: 21/50 = 42% vs 42%**. → **The PRM is a per-step
PROCESS improver, NOT an episode-OUTCOME improver.** The +14pp goal edge at n=5 was noise.

### 5.7 LLM memory & proposer-prompt experiments (`outputs/stage2_improvement_{memory,proposer,proposer_generic}.json`)
Investigated the LLM's per-box memory (it's coarse: a 3-step bool-only window with `evidence` hard-coded
empty + the exhausted-type set). Built two leak-free toggles in `engagement.py` (default OFF =
back-compat), A/B'd 8 boxes × 4 trials, episode-clustered:
- **`rich_memory`** (real trace evidence + ordered last-6 trace + per-type no-progress counts to the LLM):
  consecutive-repeat 53%→27% (p≈0), goal 12%→44% (**p=0.004**), exploit_proposed 41%→75% (p=0.007). **A
  clean, fair win — no leakage.** (This corrected an earlier prediction that memory wouldn't help success.)
- **`enhanced_prompt`** (fingerprint-once + named CVE techniques): goal 16%→53% (p=0.002) — BUT the
  **generic control** (`proposer_generic`: same strategy, CVE names removed) does NOT lift goal (19% vs
  28%, NS). → the enhanced gain was **test-set LEAKAGE** (the names matched the test boxes). `SYS_ENHANCED`
  is marked leakage-demo-only in code; only the leak-free strategy (which reduces spinning, not success)
  is keepable.
All three treatments significantly reduce spinning. Only `rich_memory` fairly lifts success (n=32, noisy
— wants higher-N replication before headlining).

### 5.8 Drupal adapter fix — pseudo-ceiling vs real ceiling
Drupalgeddon2 was 0/4 in every arm. Live diagnosis: the η recipe **fires the RCE correctly**
(`uid=33(www-data)`). Two adapter bugs, both fixed (leak-free, general):
1. η used PHP `exec()` which returns only the LAST line → `cat /etc/passwd` gave `_apt:x:100:…`, never
   `root:x:0:0`. → recipe switched to `passthru` (full output).
2. φ `_parse_fileread` matched only literal `root:x:0:0`. → generic `_PASSWD_LINE` regex (`name:x:uid:gid:`),
   robust to `exec()`-truncation on any box.
After fix: deterministic proposer solves Drupal **3/3**. BUT LLM-autonomous still **0/4** (3/4 die at step
1 with `exploit_proposed=False`). → fix removed the **pseudo-ceiling** (adapter) and isolated the **real
ceiling** (proposer doesn't propose the exploit). ψ exploit coverage also extended (deploy/write/PUT a
JSP/WAR → file_upload; trigger OGNL/RCE/deserialization → exploit_attempt; held-out false-accept still 0).

### 5.9 Three new Vulhub boxes (2026-06-21) → 15-box set
Added `tomcat/tomcat8` (8094, weak-pw manager → war-deploy RCE **to root**, ②⑤⑥), `httpd/CVE-2021-41773`
(8095, path-traversal file-read ④ + mod_cgi RCE ⑥), `nginx/insecure-configuration` (8097, off-by-slash
file-read ①④, **file-read-milestone** — no RCE). All wired + payloads verified live; deterministic proposer
solves them end-to-end (tomcat8 → uid=0 root). **Honest A/B result (deepseek-chat, 5 trials, budget 16 /
patience 6):** the autonomous LLM reaches **goal 0% on all 3** — these multi-step chains are *harder* than
the single-step boxes (tomcat8 → 0 milestones: the LLM never sequences auth→upload→RCE; httpd/nginx → ~20%
file-read, no full goal). The deterministic proposer succeeds (recipes work), so this is the **proposer
ceiling again, stressed harder by multi-step**. 15-box cluster-robust re-analysis: pooled per-step PRM
advantage over all full-goal boxes **stays significant** — Δ+12.0pp, permutation **p=0.02**, cluster-boot
CI [+3.3,+21.6]pp (was p=0.013 at 12 boxes; the 3 hard boxes dilute but don't overturn it). self-adv subset
per-step now NS (p=0.12) — diluted by the 2 hard self-adv boxes where neither arm progresses.

### 5.10 XBEN / XBOW benchmark (in progress — see also task notes)
Cloned `github.com/xbow-engineering/validation-benchmarks` to `E:\PT\xben\`; the 6 target challenges
(XBEN-022/023/029/063/089/092-24, all level-2 multi-step: SSTI/blind-SQLi/deserialization/business-logic)
are present, ports remapped to 8100-8105 (loopback). **These are canary-marked HELD-OUT eval data — never
persist the flags.** Base images pulled via mirror (Docker Hub fails); aliyun apt + tsinghua pip injected
into the 10 Dockerfiles to get past the China-network build wall; all **6/6 build + run**. Autonomous
descriptors (recon-only η, **no baked exploit**) in `stage2/targets/xben/`; runner `stage2/xben_autonomous.py`.
**RESULT (mode llm_only + efficiency stack, 3 trials, `outputs/stage2_xben_autonomous.json`): flag 0/18 = 0%**
across all 6, mean **~5.2 steps to stall** (all via `no_progress_stuck` — clean give-up, the Tier-3 breaker
working), `exploit_proposed` **100%** (the vocab hint makes it propose an exploit) but milestone only
~1.17/3 — it cannot CONSTRUCT/execute the box-specific multi-step exploit (SSTI / blind-SQLi /
deserialization / business-logic) because η has no recipe for it and the generic templates don't fire these.
**Confirms the architectural boundary:** the 16-action abstraction + per-target-baked-η design is for *known*
exploits; XBEN's *autonomous novel multi-step exploitation* is beyond it. Full XBEN coverage needs either
solving each challenge (defeats the benchmark) or a tools-image / raw-request agent — out of scope here.

---

## 6. Script map (everything in `stage2/`)

| script | what it does | run |
|---|---|---|
| `engagement.py` | the autonomous loop + proposers + rerank modes; `run_engagement(...)`. Key params: `mode`, `rich_memory`, `rerank_seed`, `shuffle_candidates`, `permissive_guard`, `patience` (global no-progress circuit-breaker, default 4, `--patience 0` disables). | `python -m stage2.engagement --executor live --proposer target --mode ab --confirmed-isolated` |
| `phi.py` | real-output → AbstractWebState parser | (library) |
| `psi.py` | enhanced ψ normalizer (Stage-2 recovery layer) | (library) |
| `eta.py` | η renderer + LiveExecutor/DryRunExecutor + `load_target` | (library) |
| `safety.py` | AuthorizationGate, lab-scope check, audit log | (library) |
| `live_ab_trials.py` | multi-trial PRM-vs-llm_only A/B on ONE box; arm-order randomization, run_metadata | see §2 example |
| `aggregate_multibox.py` | pool per-box A/B JSONs → `stage2_multibox_aggregate.json` (naive z-tests; superseded by stats_analysis) | `python -m stage2.aggregate_multibox` |
| `stats_analysis.py` | **cluster-robust** re-analysis (permutation + bootstrap + GEE + effect sizes + failure taxonomy) | `python -m stage2.stats_analysis` |
| `ablation_rerank.py` | reranker-isolation ablation; `--proposer target|llm`, `--modes`, `--boxes` | see §8 |
| `paired_ab.py` | **PAIRED** reranker A/B (shared-candidate CRN via `CachingProposer`) — variance-reduced, ~half the LLM calls; reports cache hit-rate + clustered permutation. **Built but not yet run formally.** | see §7 |
| `improvement_ab.py` | A/B two engagement configs; `--compare memory|proposer|proposer_generic` | see §8 |
| `seven_dim_report.py` | regenerate `STAGE2_SEVEN_DIM_TABLE.md` from the JSONs | `python -m stage2.seven_dim_report` |
| `reset_target.py` | reset/start/healthcheck via `target_registry.json` | `--all --check` / `--label X` |
| `preflight.py` | offline readiness + live-infra (docker/containers/healthchecks) | `python -m stage2.preflight` |
| `replay.py`/`closed_loop.py`/`fixtures.py`/`eval_psi.py` | Phase-1 offline replay, ψ held-out eval | `python -m stage2.replay`, `python -m stage2.eval_psi` |
| `live_smoke.py` | fixed read-only recon→RCE sequence on one box (sanity) | gated |

Artifacts: `outputs/prm_strong.joblib` (the frozen PRM). Reports (top level): `STAGE2_LIVE_RESULTS.md`
(the canonical results doc, §1–§6 + superseded appendix), `STAGE2_SEVEN_DIM_TABLE.md`,
`STAGE2_INFERENCE_METRICS.md`, `STAGE2_PLAN.md`, `STAGE2_PHASE2_RUNBOOK.md`, `STAGE2_ENVIRONMENT.md`.

---

## 7. Reproduce the experiments (exact commands)

All from `$ROOT`. Prefix live LLM commands with the two env vars from §2.

```powershell
# health of all 12 boxes (non-destructive)
python -m stage2.reset_target --all --check

# 12-box A/B already in outputs/ ; re-aggregate (naive) and the cluster-robust re-analysis:
python -m stage2.aggregate_multibox
python -m stage2.stats_analysis
python -m stage2.seven_dim_report

# reranker-isolation ablation, key-free (deterministic proposer):
python -m stage2.ablation_rerank --proposer target --seeds 8 --budget 12 --executor live --confirmed-isolated

# reranker-isolation ablation, REAL LLM (needs DEEPSEEK_API_KEY):
python -m stage2.ablation_rerank --proposer llm --model deepseek-chat --seeds 5 --budget 12 `
  --boxes ThinkPHP-5-rce ThinkPHP-5.0.23 Struts2-S2-048 Struts2-S2-045 Joomla-8917-sqli php-cgi-2012-1823 `
  --modes llm_only random prm oracle --executor live --confirmed-isolated `
  --report-output outputs\stage2_ablation_rerank_llm.json

# memory & proposer experiments (needs key):
python -m stage2.improvement_ab --compare memory          --trials 4 --executor live --confirmed-isolated
python -m stage2.improvement_ab --compare proposer        --trials 4 --executor live --confirmed-isolated
python -m stage2.improvement_ab --compare proposer_generic --trials 4 --executor live --confirmed-isolated

# PAIRED reranker A/B (variance-reduced; built, NOT yet run). Offline self-check (no key):
python -m stage2.paired_ab --proposer target --executor dryrun --trials 2 --budget 8 `
  --boxes ThinkPHP-5-rce Struts2-S2-048 --modes llm_only prm random oracle --confirmed-isolated
# live formal run (later, needs key): swap --proposer llm --model deepseek-chat --executor live

# tests
python -m pytest tests\ -m "not slow" -q
```

Long LLM runs (1–2h): launch in the background and poll the output file. They have per-trial try/except
so transient DeepSeek SSL/EOF blips degrade N rather than abort.

---

## 8. Conventions you MUST follow

1. **Honesty / anti-inflation.** Report negative results, leakage, and noise. When a result looks good,
   attack it with a control (a generic-prompt control, a random-rerank control, a higher-N replication,
   a cluster-robust test). Keep only what survives. Quote p-values and CIs, say "NS" when NS.
2. **Statistics:** for any per-step (high-N, correlated) metric, use the **episode-clustered permutation
   test** (`stats_analysis.stratified_permutation`), NOT a naive two-proportion z-test. Report effect
   size + CI + clustered p + significance, per-box AND pooled. Separate auth-milestone boxes.
3. **No leakage:** never inject the box's known vuln/payload into φ or the proposer prompt. Public-CVE
   retrieval (what a human would google) is fair; the descriptor's specific answer is not. The
   `SYS_ENHANCED` prompt is a *leakage demo* — do not use it as a real result.
4. **PRM features stay on training-time Stage-1 ψ.** Never feed oracle q-values as PRM features. Don't
   swap the PRM's normalizer without retraining.
5. **Schema frozen** (16 actions). ooa is 8.5% — schema is not the bottleneck.
6. **Safety:** live execution only against the owned `127.0.0.1` boxes, gated + audited, read-only
   commands (`id`/`whoami`/`cat /etc/passwd`). Both env var + `--confirmed-isolated` required.
7. **Key hygiene:** `DEEPSEEK_API_KEY` in env only; never write/print it. Tell the user to rotate after.
8. **Keep tests green** (393/1-skip) and update `STAGE2_LIVE_RESULTS.md` + the auto-memory when metrics
   change.

---

## 9. Gotchas (will bite you)

- **PowerShell vs Bash:** the Bash tool is POSIX (`$null` redirect is `2>$null`-incompatible — use
  `/dev/null`); PowerShell uses `;`/`if($?)`, no `&&`. Don't mix. For Go-template `docker inspect`,
  quoting breaks in PowerShell — parse `docker inspect <c> | ConvertFrom-Json` instead.
- **GBK encoding:** Windows console is GBK; emoji/some unicode in `print()` crash with
  `UnicodeEncodeError` even when the file write (utf-8) is fine. Avoid emoji in stdout; LiveExecutor
  already decodes subprocess bytes as utf-8/replace.
- **curl globbing:** `[]` in payloads needs `curl -g`.
- **Docker Hub pulls fail** (TLS/EOF). Use `docker pull docker.m.daocloud.io/vulhub/<img>` in a
  resume-retry loop, then `docker tag` to `vulhub/<img>`.
- **PHP `exec()` returns only the LAST line** of multi-line output (the Drupal bug, §5.8). Use
  `passthru`/`system` for full file reads. φ's `_PASSWD_LINE` now tolerates partial passwd reads.
- **per-episode goal-reach is NOISY** at n≤32 (baselines drifted 12–28% across runs). Trust per-step
  (high-N) + clustered tests; replicate goal-reach at higher N before claiming.
- **`deepseek-v4-pro` is ~6h/box** — don't use it for multi-box studies; use `deepseek-chat`.

---

## 10. Open work — what to do next (ranked, actionable)

The headline open problem is the **`exploit_never_proposed` ceiling**: on some boxes the LLM proposer
never puts a foothold-class action on the table (Drupal: 3/4 die at step 1; php-cgi: non-self-advertising,
1/8). Memory/reranking/adapter fixes cannot touch this — it is a **proposer-capability** problem. Diagnose
each stuck box first (is it *proposed-but-fails* = adapter, or *never-proposed* = proposer?) using the
deterministic-proposer goal check (`mode='oracle'`, TargetAwareProposer) — if it reaches goal, the adapter
is fine and the gap is the proposer/ψ.

**Experiment randomness / variance — current handling + the open fix.** Each box runs **5 trials/arm**
(10 in the high-N extension), NOT once; the LLM (temp 0.5) stochasticity is what the multiple trials
sample, both arms share the proposer (fair), and `stats_analysis` uses fixed analysis seeds (12345) with
cluster-robust tests. **Known limitations:** (a) the live A/B is **unpaired** — `prm` and `llm_only` are
independent stochastic rollouts, so proposer-draw luck adds variance; (b) the DeepSeek sampling is not
seed-pinned (not bit-reproducible); (c) n=5 is small for the noisy per-episode-goal metric. The
**paired-A/B design (`stage2/paired_ab.py` + `CachingProposer`) is now BUILT** to remove (a) — it shares
the LLM candidate set across arms wherever states coincide (CRN), so the only per-step difference is the
ranking, and it ~halves LLM calls. **It has NOT been run formally yet** — running it is item 0 below.

Ranked next steps (all leak-free):

0. **Run the paired A/B** (`python -m stage2.paired_ab --proposer llm --model deepseek-chat --executor
   live --confirmed-isolated --trials 5`). It is built + offline-smoke-tested; this is the variance-reduced
   re-measurement of the per-step PRM-vs-baseline result (and `prm` vs `random`/`oracle`). Report the cache
   hit-rate (how much pairing was achieved) alongside the clustered permutation p-values; compare to the
   unpaired numbers in §5.5.
1. **Higher-N replication of `rich_memory`** (cheap, closes a known gap). The fair success win (12%→44%,
   p=0.004) is at n=32 and goal-reach is noisy. Run `improvement_ab --compare memory --trials 10` on the
   self-adv boxes; report whether the lift holds under clustering.
2. **Stronger-model probe** for the proposer ceiling. Run the LLM ablation / a small A/B with
   `--model deepseek-reasoner` on the never-proposed boxes (php-cgi, Drupal) vs `deepseek-chat`; measure
   `exploit_proposed` rate + goal. Tells you how much of the ceiling is just model capability. (reasoner
   is slower — keep trials small, watch the clock.)
3. **RAG-over-public-CVE proposer** (the principled fair fix). Give the LLMProposer a retrieval step over
   a GENERIC public corpus (ExploitDB / Metasploit module list / CVE-DB) keyed on the φ fingerprint, so
   it can recall "stack X → known technique Y" the way a human googles — WITHOUT the descriptor's specific
   answer. A/B vs no-RAG; measure `exploit_proposed` + goal. This is the honest version of the (rejected)
   CVE-name cheat-sheet.
4. **Tools-image discovery** (most realistic; needs operator infra). Build a tools image with
   `searchsploit`/`nuclei`/`nmap --script` so the agent DISCOVERS the vuln from tool output. This is the
   long-standing Phase-3 blocker (host has only curl/bash). Then the discovery is leak-free and real.
5. **ψ precision** for `proposed-imprecisely` boxes (Tomcat). Extend `EnhancedNormalizer` exploit/upload
   coverage further; keep held-out false-accept = 0 (`python -m stage2.eval_psi`).
6. **More complete-chain boxes** (DVWA — `stage2/targets/dvwa.json` already drafted; Juice Shop; an
   upload-RCE) to widen coverage beyond the current 12.
7. **Make `rich_memory` the default** in the production loop if step 1 confirms it (it's a clean win).

**Research frontier (out of scope for the PRM):** autonomous construction of a *novel* multi-step exploit
for an unknown vuln. The PRM is a value/ranking model, not an exploit generator. Don't expect
memory/reranking to solve it.

---

## 11. Glossary

- **PRM** — Pentest process-reward model; a per-step *reranker* of proposed actions, `prm_strong.joblib`.
- **φ / ψ / η** — observe (real output→state) / normalize (text→16 actions) / render (action→command).
- **per-step progress** — fraction of steps that produced new abstract information; the high-N HEADLINE.
- **self-advertising** — the stack reveals itself in the fingerprint (so the LLM can name the CVE).
- **auth-milestone box** — goal unreachable by a single command by design (WebLogic, Gitea).
- **exploit_never_proposed** — the proposer never emits a foothold-class action; the dominant ceiling.
- **leakage** — telling the agent the answer it should discover (e.g. the box's CVE technique in the prompt).
- **pseudo-ceiling vs real ceiling** — adapter bug that *looks* like a model limit, vs a true proposer
  capability limit.

---

**Start here:** run `python -m stage2.preflight` and `python -m stage2.reset_target --all --check` to
confirm the 12 boxes are up, `python -m pytest tests\ -m "not slow" -q` to confirm green, then pick item
#1 from §10.
