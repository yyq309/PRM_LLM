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

E.7 then asks whether the findings depend on which LLM we use (they do not), and E.8 reports the one honest
limitation.

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
which the single-service boxes cannot.

**How we measure, and the fair comparison.** Every real-target result is a head-to-head between two agents
that use the **same LLM** and differ in only one thing:
- **`prm`**: the LLM proposes actions, and the advisor re-ranks them;
- **`llm_only`**: the LLM's own ordering is used, with no advisor.

So any difference is attributable to the advisor alone. We report:
- **goal-rate / root-rate** — the fraction of attempts that actually reached the goal / got root (unambiguous,
  cannot be gamed);
- **per-step progress** — the fraction of *steps* that made forward progress rather than being wasted;
- **ranking accuracy** — how often the advisor's top-ranked action is genuinely the best one.

Because a run is a sequence of correlated steps (not independent coin flips), naive statistics overstate
significance. We use **episode-clustered permutation tests** and **bootstrap confidence intervals**, pair the
two agents on identical situations (**common random numbers**), and apply a **multiple-comparison correction**.
We report these conservative numbers throughout.

## E.3 Question 1 — Does the simulator teach good judgment?

**In one sentence: yes — the advisor learns a real (if modest) sense of which action makes progress, and it
does so without any leakage.** Three pieces of evidence:

**(a) It ranks progress correctly.** We compare the value oracle against a mathematically *optimal* solver of
the simulator (computed by value iteration). When the optimal solver is told to value only *true progress
toward the goal*, the oracle agrees with it on the **best action 74 % of the time**, puts the best action in
its **top 3 every time (100 %)**, and its rankings correlate with the optimum at **Spearman +0.45** (where 0
is random and 1 is perfect). In short, the simulator-trained value is a genuine progress signal, not noise.

**(b) The advisor itself is a good ranker.** A standard way to score a ranker is **pairwise accuracy**: given
two actions where we know which is better, how often does the advisor rank them correctly? (0.5 is a coin
flip; 1.0 is perfect.) The advisor reaches **0.94** on seen-type tasks, **0.98** on new instances of seen
chains, and — the demanding test — **0.92 on entirely new attack-chain shapes it never trained on**. It also
reliably flags actions that should *not* be taken (detecting bad actions at **AUC 0.93**, where 0.5 is
chance). The advisor keys on the *situation*, not on having memorized a specific attack, which is why it
generalizes to unseen chain shapes.

**(c) No cheating.** An audit confirms the advisor's input contains **no hidden answer** — no secret path,
credential, or flag — and that hiding any single observable field degrades it only gracefully. Its skill
comes from observable context, not leaked secrets.

One honest boundary: the advisor is a **ranker, not a player**. If we let it drive an agent entirely on its
own it does not succeed as a standalone policy — its value is in *advising*, which is exactly how we use it.

## E.4 Question 2 — Does the advice transfer to real targets?

**In one sentence: yes — on real machines, letting the advisor re-rank the LLM's actions produces a
statistically significant improvement in per-step decision quality.** First, the adapter works: the
interpreter maps real LLM text to the right action type **92 %** of the time (and an improved version raises
the hardest cases from 49 % to **78.5 %**). Then, across the 15 real web boxes, the `prm` agent makes
forward progress on more of its steps than `llm_only`, and the gap is significant under the conservative,
clustered statistics: **p = 0.02** (still significant after multiple-comparison correction). Plainly: a
ranking sense learned in a cheap simulator, with zero real labels, measurably improves real per-step action
choices. (We are careful to distinguish *per-step* quality, which clearly improves, from *whole-episode*
success on these single-service boxes, which is often tied — we return to why in E.5 and E.8.)

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
the LLM's own instincts are weakest — local privilege escalation — which only a full-machine target exercises.**

The second VM, **Toppo**, draws a clean boundary: *both* agents fail autonomously because the LLM never even
proposes the needed "find credentials → SSH in" step — yet a scripted (non-LLM) agent reaches root on both
machines. So the adapter and the advisor are sound; the failure is a *limit of the LLM's imagination*
(it cannot rank an action it never proposes), not a broken transfer.

## E.6 Question 4 — Where does the advisor break, and why?

**In one sentence: the advisor systematically over-values reconnaissance, and we show this is a built-in
consequence of how it was trained — not a bug we could patch.** In the simulator, training rarely creates
situations where the agent already knows everything but keeps scouting; as a result the advisor learns that
"reconnaissance" is almost always valuable. Concretely, its average score for the action "enumerate web
paths" is **0.89**, far above "run the exploit" at **0.54**. On real targets this shows up as the advisor
adding scouting steps a capable LLM does not need.

The important part is that this resists repair. We tried **three independent fixes** — down-weighting recon at
inference time, re-labelling the training data, and forbidding recon when better actions exist — and **all
three failed** to remove the bias without damaging the advisor elsewhere. A multi-seed check further shows the
bias size is itself unstable across training seeds. We therefore present this not as a defect to fix but as a
**characterized, structural limitation of simulator-to-real value transfer** — a transferable warning for
anyone training advisors in a simplified world: whatever situations the simulator under-represents, the
advisor will mis-value in reality.

## E.7 Does the result depend on which LLM?

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

So outcome-help is *conditional* (it appears when the LLM is weak), ranking-help is *near-universal*, and the
per-step effect tracks how long the attack is — one consistent mechanism, across three vendors, **not specific
to any single model.**

## E.8 The one honest limitation

We state plainly — and choose *not* to build the paper around — the following: the advisor's benefit on the
*final outcome* depends on how good the LLM's own ordering already is. It clearly helps a weak or
un-coached LLM (per-step **+12 points**, p = 0.02), but if we *coach* the LLM with an explicit hint about the
action vocabulary, that coached LLM can do better *without* the advisor than with it (66.7 % vs 39.6 % in that
configuration). We report this efficiency reversal in full rather than hiding it. We do not center the paper on
it because (a) the "coached" LLM there used an author-supplied hint, which confounds the comparison, and
(b) it is a specific case of a phenomenon already known in the literature (a model that *checks* work being
obviated by a generator that no longer makes the mistakes).

## E.9 Summary

The cheap abstract simulator produces a genuine, leak-free sense of which action makes progress (E.3); that
sense **transfers** to real machines and significantly improves per-step choices (E.4); it **drives complete
real attacks to root**, earning its keep precisely in the hardest privilege-escalation phase (DC-1: 100 % vs
56 %; E.5); it has a **clearly characterized failure mode** — over-valuing reconnaissance, which resists three
fixes (E.6); and the entire picture **reproduces across three different LLMs** under one simple rule (E.7).
The advisor's effect on final success is *conditional* on the LLM being weak, which we report honestly as a
limitation (E.8).
