# Stage 2 — Phase 1 Report (offline φ/ψ replay, no live execution)

**What this measures.** How well the frozen 16-action abstraction + Stage-1 ψ normalizer + the new φ
parser cover the *real* attack chains of representative VulnHub-class single-host boxes, and whether the
abstract-trained Pentest-PRM reranks real candidate actions. NOTHING here executes against a target —
it replays recorded tool output through the adapter. This is the decisive, cheap, safe step that gates
whether (and where) the schema needs extending before any live Phase-2 work.

## Headline numbers

- **Boxes:** 7  |  **Total real steps:** 71
- **Out-of-abstraction rate (ground-truth, hand-labeled): `8.5%`**
- **ψ normalizer accuracy on in-abstraction steps:** `49.2%`  (false-reject `32.3%`)
- **φ field recall (state reconstruction vs hand-labels):** `94.8%`
- **PRM rerank top-1 (good vs distractors):** `50.0%`
- **Offline closed-loop chain-adherence (PRM-autonomous proxy):** `38.4%`

## Key finding — the bottleneck is ψ, not the schema

The decisive Phase-1 result **inverts the expected risk**. Going in, the open question was whether
the frozen 16-action abstraction is too small for real boxes. It is not:

- **Abstraction coverage is high:** only `8.5%` of real steps fall outside the 16 actions (hand-labeled ground truth). The schema structurally covers ~92% of real chains.
- **The normalizer (ψ) is the real wall:** on the in-abstraction steps — the ones that *do* have a
  correct abstract action — the Stage-1 ψ maps only `49.2%` correctly, false-rejecting `32.3%`
  of them outright. Natural operator phrasing is messier than ψ's synthetic training intents.

So the Phase-2 prerequisite is **ψ normalizer coverage, not schema extension**. The schema-gap
shortlist below is small and entirely *non-web* primitives (SSH login, offline hash cracking, local
account switch, a binary/SMB service exploit) — genuinely out of a single-host-**web** scope, not
evidence the web schema is incomplete.

**Where ψ false-rejects in-abstraction intents (by hand-labeled action):**

- `sensitive_file_read` — 11 miss(es)
- `vulnerability_check` — 2 miss(es)
- `command_execution` — 2 miss(es)
- `service_enumeration` — 1 miss(es)
- `web_path_enumeration` — 1 miss(es)
- `content_retrieval` — 1 miss(es)
- `post_exploitation` — 1 miss(es)
- `input_discovery` — 1 miss(es)
- `auth_attempt` — 1 miss(es)

**Where ψ routes to the wrong action (defensible boundary cases vs hard errors):**

- `post_exploitation -> privilege_escalation` — 4
- `sensitive_file_read -> content_retrieval` — 3
- `vulnerability_check -> exploit_attempt` — 1
- `auth_attempt -> input_discovery` — 1
- `exploit_attempt -> file_upload_attempt` — 1
- `exploit_attempt -> vulnerability_check` — 1
- `web_path_enumeration -> input_discovery` — 1

> Honesty note: these clusters are measured on author-constructed fixtures, so they show ψ's coverage
> shape, not a population rate. A real ψ-coverage fix must be validated on **held-out** recorded
> operator logs (STAGE2_PLAN §3) — tuning ψ against these same fixtures would be in-sample and
> circular. We report the gap; we do **not** report a flattering in-sample 'after' number.

## ψ coverage layer (Stage-2) — bottleneck addressed, honestly

`stage2/psi.py` adds a Stage-2-local coverage layer over the **frozen** Stage-1 normalizer (Stage 1
is untouched, so the oracle/PRM artifacts stay reproducible). It only acts when Stage-1 returns
`unsupported`: an **out-guard** keeps non-web primitives (SSH, offline crack, su, kernel/binary
overflow) out, then a general verb×object recovery layer maps the in-scope intents the keyword
matcher missed. The recovery vocabulary is tuned **only** on an independent benchmark
(`stage2/psi_benchmark.jsonl`, 124 labeled intents) — *disjoint from these fixtures*.

**Generalization is measured on the 7 walkthrough fixtures as a HELD-OUT test** (the layer never saw
their strings):

| set | ψ | accuracy | false-reject | false-accept |
|---|---|--:|--:|--:|
| dev benchmark (tuned-on) | Stage-1 | 33.9% | 53.6% | 33.3% |
| dev benchmark (tuned-on) | enhanced | 75.0% | 12.5% | 0.0% |
| **held-out fixtures** | Stage-1 | 49.2% | 32.3% | 0.0% |
| **held-out fixtures** | **enhanced** | **78.5%** | **3.1%** | **0.0%** |

On the held-out fixtures, ψ accuracy rises **49% → 78%**, false-reject collapses **32% → 3%**, and
**false-accept stays 0** (the out-guard never maps an out-of-abstraction step to a web action).
Because this is held-out, the lift is genuine generalization — not in-sample fitting.

**Train/inference coupling (important):** the frozen strong PRM was trained on Stage-1-ψ features, so
the PRM candidate pool is intentionally still normalized by the **training-time** ψ. With that, the
rerank top-1 is preserved at `0.5`. A diagnostic that *also* swaps the
PRM's candidate normalizer to enhanced-ψ degrades rerank top-1 to `0.1` (out-of-distribution for the
frozen model) — confirming the coverage layer must drive η/execution, not the PRM's feature path,
unless the PRM is regenerated and retrained on enhanced-ψ features (a Stage-1 retrain).

> Still honest about limits: 78.5% is on 65 author-constructed in-abstraction steps, not live logs;
> the residual misses are mostly defensible action-boundary cases (post-exploitation vs
> privilege-escalation, config-read as content_retrieval vs sensitive_file_read). The remaining gap
> and the PRM-retrain question both still want **held-out real operator logs** to settle.

## Decision gate

> Threshold: out-of-abstraction > **60%** ⇒ schema-extend before Phase 2.
>
> **Verdict:** Abstraction covers the bulk of the real chains (out-of-abstraction 8.45% <= 60%); proceed to Phase 2 (η + single-box loop).

**Schema-extension shortlist (prioritized by frequency across boxes):**

- `offline_credential_cracking` — 2 occurrence(s)
- `ssh_remote_login` — 2 occurrence(s)
- `binary_service_exploit` — 1 occurrence(s)
- `local_account_switch` — 1 occurrence(s)

## Per-box breakdown

| Box | Family | Steps | Out-of-abs | ψ acc | φ recall | PRM top-1 |
|---|---|--:|--:|--:|--:|--:|
| DC-1 | rce_privesc | 10 | 10% | 78% | 100% | 33% |
| Kioptrix-L1 | rce_privesc | 7 | 14% | 33% | 89% | — |
| Mercury | leak_authed_privesc | 10 | 10% | 56% | 100% | 100% |
| Mr-Robot | default_or_weak_password | 15 | 13% | 38% | 83% | 100% |
| Raven-2 | chained_exploit | 11 | 0% | 45% | 100% | 0% |
| SkyTower | leak_authed_privesc | 8 | 12% | 43% | 100% | 0% |
| Stapler | default_or_weak_password | 10 | 0% | 50% | 94% | 100% |

## Honest reading

- The **out-of-abstraction rate** is the make-or-break number: it is the fraction of real steps that
  have *no* representation in the 16-action web schema (SSH logins, SMB/binary service exploits,
  kernel/offline-crack, SSTI/SSRF/XXE, vhost/subdomain enumeration). It is hand-labeled ground truth,
  not inferred from ψ, so ψ errors cannot flatter it.
- **ψ accuracy < 100%** on in-abstraction steps is expected: real operator phrasing is messier than the
  synthetic training intents; the gap is a normalizer-coverage TODO, reported not hidden.
- **φ recall < 100%** reflects genuinely lossy parsing of messy real output.
- The **PRM is a reranker, not an autonomous policy** (Stage-1 finding, reconfirmed here): its score head
  has an enumeration bias, so the closed-loop chain-adherence proxy is modest. Its intended use is to
  rerank a small LLM-proposed candidate set, where the rerank top-1 metric applies.
- These fixtures are **author-constructed from public write-up structure**, not live captures. They
  measure schema/normalizer/parser coverage, which is structural; live value-uplift is Phase 2/3 and
  remains gated behind an owned, isolated, authorized lab.
