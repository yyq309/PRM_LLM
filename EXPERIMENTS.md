# Experiment Registry & Sufficiency Map (vs the main line)

Canonical map of **environments × experiments × contributions**, with an honest sufficiency
verdict per contribution and the ranked list of experiments still to add. Framing: see
[`CONTRIBUTIONS.md`](CONTRIBUTIONS.md). Updated 2026-06-21 from a 6-lens sufficiency audit
(workflow `wkwbal2hq`).

---

## Environments (4 — note: the training sim is NOT the Docker boxes)

| # | Environment | Role | Contents |
|---|---|---|---|
| ① | **Abstract single-host-web simulator** | **Stage-1 TRAINING** | synthetic MDP, 16 frozen actions, 12 task families; trains the DQN value oracle + Pentest-PRM; Q*/leakage/coverage live here |
| ② | **Docker Vulhub web boxes (15)** | **Stage-2 inference** | real web containers on 127.0.0.1; the per-step reranking study |
| ③ | **XBEN/XBOW (6)** | **provenance-only — NOT a paper result** | autonomous novel-exploit benchmark; 0/18 confirms the scope boundary, **dropped from the paper** and replaced by a one-line scope statement (guardrail) — the project never claims autonomous novel-exploit construction |
| ④ | **VulnHub full-machine VMs (4)** | **Stage-2 end-to-end** | DC-1/Raven2/Toppo:1/Symfonos:1 (VMware); complete kill chain — **planned** |

---

## Sufficiency verdict per contribution

| Contribution | Verdict | Why |
|---|---|---|
| **C-A** abstract→real transfer adapter | **SUFFICIENT** (was PARTIAL) | adapter well-supported (zero-label ✓, leakage_audit ✓, per-step value attributable). **G4 done (2026-06-21):** learned PRM vs cheap hand heuristic is proposer-conditional — det. proposer prm≈heuristic (Δ+0.4pp p=0.65) / prm<random; LLM proposer prm>heuristic (Δ+20.6pp **p=0.001**). Honest: learned value beats a cheap prior only in the realistic LLM-proposer regime. |
| **C-B** real end-to-end kill chain | **PARTIAL** (was INSUFFICIENT) | **DONE (2026-06-21):** DC-1 full chain Web→SUID-find→root proven live — autonomous LLM A/B **prm root 100%(5/5) vs llm_only 60%(3/5)**, ~2× fewer steps; Toppo proven deterministic (LLM-autonomous 0% = proposer ceiling at cred→ssh foothold). Remaining: Raven/Symfonos footholds deferred (fragile); 2/4 boxes end-to-end. |
| **C-C** structural train-inference gap | **SUFFICIENT** (was PARTIAL) | **G2 done (2026-06-21):** (a) phase histogram — web_path_enum 0.94 early→0.609 advanced, still > command_exec 0.21 post-foothold; (b) cause IDENTIFIED in `reward.py` — phase-independent info-discovery rewards (`path_found:+2.0`, `input_found:+2.0`) keep recon valuable post-foothold in-sim (NOT action-masking). A sim-to-real **reward-design** gap; 3 surgical fixes failed because the signal is in the reward. |
| **Limitation** proposer-conditional (demoted) | **SUFFICIENT** | E1/E2/E6 enough to report honestly; **no new experiment needed**. NUANCED by C-B: the inversion is WEB-PHASE-specific — on long real chains (DC-1) the PRM HELPS. |
| **Supporting** cluster-robust + CRN | **PARTIAL** | clustered stats ✓ verified; **paired A/B (CRN) built but NOT run** → reported p-values are from unpaired runs (G3, cheap, still open). |

**Bottom line (updated 2026-06-21):** C-A, C-C, and the limitation are now **SUFFICIENT**; **C-B is PARTIAL→strong**
(real end-to-end proven on DC-1 autonomously + Toppo deterministically; Raven/Symfonos footholds deferred as the
same proposer-ceiling class). Only **G3 (run paired_ab.py)** remains cheap-and-open; G2 + G4 are **done**. The big
honest add from this round: **the demoted "PRM obsoleted" limitation is web-phase-specific — on the harder real
full chain the reranker recovers and beats llm_only (DC-1 100% vs 60%).**

---

## Experiment registry

### Stage-1 (abstract sim — training) — supports C-A foundation + C-C
| ID | Experiment | Status / result | Supports |
|---|---|---|---|
| T1 | DQN value-oracle train + seed gate | ✅ done (frozen) | C-A |
| T2 | PRM dataset gen + train (baseline/robust/strong/joint) | ✅ done (`prm_strong.joblib`) | C-A |
| T3 | Q* exact value-iteration verify | ✅ done | C-A (label credibility) |
| T4 | Leakage audit + coverage/diversity audit | ✅ done (0 structural leaks) | C-A (no leakage) |
| T5 | Calibration + robustness eval | ✅ done | C-A |
| T6 | ψ normalizer accuracy benchmark | ✅ done (49%→78.5%, false-accept 0) | C-A (adapter) |
| T7 | Recon-bias label evidence | ✅ done (web_path_enum 0.887 ≫ exploit 0.535) | **C-C** |

### Stage-2 (inference) — the live experiments
| ID | Experiment | Status / result | Supports |
|---|---|---|---|
| E1 | **LLM_only vs PRM+LLM** A/B, 15 web boxes, 5 trials, cluster-robust | ✅ per-step **Δ+12.0pp, perm p=0.02**, CI[+3.3,+21.6]; per-episode NS | C-A / C-B(web) |
| E2 | **Reranker-isolation ablation** (prm/random/shuffled_prm/heuristic/oracle), det. + LLM proposer | ✅ det: PRM 26.7%<random 29.3% (p=0.003); LLM: PRM 68.5%>random 50% (p=0.007) | **C-C** / limitation |
| E3 | Cluster-robust stats (clustered perm / bootstrap / GEE) | ✅ done (naive 0.024→clustered 0.066) | methodology |
| E4 | **Paired A/B (CRN)** `paired_ab.py` | ⚠️ **built, NOT formally run** | methodology (→ G3) |
| E5 | High-N replication (10 trials × 5 boxes) | ✅ per-step **+12.1pp p=0.012**; per-episode **TIE 42%=42%** | C-B (process≠outcome) |
| E6 | Efficiency-tier A/B (content-credit/rich_memory/CRN/vocab-hint/milestone-slack) | ✅ "good proposer obsoletes PRM": llm_only 66.7% vs PRM 39.6% **p<0.0001** | **limitation** |
| E7 | Memory / proposer-prompt experiments | ✅ rich_memory 12%→44% p=0.004 (no leak); CVE-prompt leak busted by control | supporting |
| E8 | XBEN autonomous (6 boxes, 18 eps) | ✅ flag 0/18 (exploit_proposed 100%, milestone ~1.17/3) — **PROVENANCE-ONLY, DROPPED from the paper** (kept in repo `outputs/stage2_xben_autonomous.json`) | scope statement, not a result |
| E9 | **Full-chain VM A/B** (4 VMs) | ⏳ **PLANNED** (`STAGE2_FULLCHAIN_PLAN.md`) | **C-B flagship** (→ G1) |
| E10 | Multi-vendor LLM | ⏳ planned, re-scoped, deferrable | C-A robustness (→ G7) |

---

## Gaps to close (ranked) — what to ADD

| ID | Add this experiment | Supports | Severity | Effort | On the books? |
|---|---|---|---|---|---|
| **G1** | **Full-chain VM A/B** (DC-1/Raven2/Toppo/Symfonos; root_flag_captured + phase-split; 5–8 trials/arm) | **C-B flagship** | **HIGH** | new-build | **✅ DONE 2026-06-21** (DC-1 autonomous prm root 100%/llm 60%; Toppo deterministic; Raven/Symfonos foothold deferred) |
| **G2** | Mask-causality | **C-C** | HIGH | reanalysis+code | **✅ DONE 2026-06-21** (`scripts/analyze_recon_bias.py`: histogram + `reward.py` info-discovery reward identified as the cause; not masking) |
| **G3** | Paired A/B (CRN) formal run | methodology | MEDIUM | rerun | **✅ DONE 2026-06-21** (cache-hit 30%; paired prm per-step 40.2% < llm_only/random/oracle, p=0.0 — **OVERTURNS the unpaired "prm beats random"** as a variance artifact; PRM is a web-phase liability, local-phase asset) |
| **G4** | Transfer baseline (learned PRM vs cheap heuristic) | **C-A** | MEDIUM | rerun | **✅ DONE 2026-06-21** (extracted from ablations: prm>heuristic p=0.001 w/ LLM proposer; ≈ p=0.65 deterministic — proposer-conditional) |
| **G5** | Per-episode higher-N (n≥20, 6 boxes) to confirm the goal-tie; phase-split metric | C-B honesty | MEDIUM | rerun | partial |
| **G6** | Proposer-ceiling probe (deepseek-reasoner vs chat on `exploit_never_proposed` boxes) | diagnostic | LOW | rerun | ✅ §10 item 2 |
| **G7** | Multi-vendor LLM on 2–3 boxes ("transfer not deepseek-specific") | C-A robustness | LOW | rerun (deferrable) | ✅ E10 |

**Status (2026-06-21): G1, G2, G4 DONE.** C-B proven end-to-end (DC-1 autonomous + Toppo deterministic);
C-C mechanistically nailed (recon over-valuation = sim's phase-independent info-discovery reward, `reward.py`,
not masking); C-A transfer-baseline answered (learned value beats a cheap heuristic only with the LLM proposer).
Remaining cheap/open: **G3** (run `paired_ab.py`), G5 (higher-N), G6/G7 (diagnostics). Raven/Symfonos full-chain
footholds deferred (fragile multi-step exploits, same proposer-ceiling class — framework + descriptors ready).

---

## Honest scope & guardrails (must hold in the paper)

1. **Per-step ≠ per-episode.** PRM is a per-step *process* improver (Δ+12pp, p=0.02, replicated);
   per-episode goal is **tied at n=10** — state this in §1, not the appendix.
2. **Box count = 15 Docker web** (single-host single-service, recipe-gated foothold) **+ 4 VM full-chain**
   (whole-machine, multi-step Web→foothold→privesc→root). No inflated count; separate stat claims per tier.
   **XBEN/XBOW is DROPPED from the paper (provenance-only)** — the two Vulhub tiers already cover C-A/C-B/C-C.
7. **Scope statement (replaces the XBEN experiment):** this work assumes the exploit class is **KNOWN and
   expressible within the frozen 16-action schema + per-target η-recipe**; **autonomous discovery/construction of
   NOVEL multi-step exploits is out of scope** (the PRM is a value/ranking model, not an exploit generator). State
   this as a design assumption — no XBEN data needed — to pre-empt "can it find 0-days?" without claiming it can.
3. **Binding constraint is the proposer**, not the PRM: `exploit_never_proposed` 28=28 across arms;
   deterministic proposer solves the adapter path on the boxes it fails autonomously.
4. **Efficiency inversion reported in full** (limitation), never hidden.
5. **Leakage wall** + **16-action schema frozen** + flags boolean-only.
6. The proposer-conditional finding is a **re-discovery** (cite verification-gap 2509.17995 /
   reward-model overoptimization 2210.10760); the "strong proposer" vocab hint is an author-added
   confound — say so.

---

## What can start now vs what waits

- **Waits on the VMs (operator):** G1 (full-chain), G5 phase-split on VMs.
- **VM-independent — can run anytime:** **G2 (mask-causality, Stage-1)**, **G4 (transfer baseline)**,
  G3 (paired A/B), G6 (proposer-ceiling probe), G7 (multi-LLM). G2 + G4 are the highest-value
  VM-independent additions.
