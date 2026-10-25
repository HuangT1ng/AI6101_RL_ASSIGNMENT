### Discussion: how action noise affects learning, and whether noisy training improves robustness

**Effect on learning.** With the template's 15,000 episodes, all five noise-free agents converge (first success after
10.7k–11.1k episodes), but **none of the ten noisy agents reaches the goal even once during training**. Their greedy policies
therefore have a success rate of 0 under every $p_{test}$. Noise breaks the optimism-driven exploration that solves the
noise-free task. The first success requires pushing the box ≈15 cells along a cliff-lined corridor. Slips near the corridor
often send the agent or the box into a cliff, and the −1000 is credited to the action that was *chosen*. Corridor states
therefore start to look worse than the safe start region before the +1000 has ever been found. With 300,000 episodes all
settings learn, but noise delays the first success from 110.8k–113.5k episodes ($p=0$) to 140.9k–161.0k ($p=0.1$) and
160.0k–219.1k ($p=0.2$). Even the optimal policy fails 41% ($p=0.1$) or 75% ($p=0.2$) of the time, because 39 consecutive
moves next to cliffs have to survive the random slips. Low success rates under noise are therefore mostly a property of the
task, not of the learner.

**Cross-noise generalisation (final policies, 1,000 episodes per cell).**
* $p_{train}=0$ is perfect at $p_{test}=0$ but reaches only 0.494 / 0.168 success at $p_{test}$ = 0.1 / 0.2. That is 84% / 67%
  of the DP policy's 0.587 / 0.252. It only knows the shortest route, which hugs the cliffs, and it rarely visited the
  off-path states that a slip leads to.
* $p_{train}=0.1$ gives the most robust policies. Four of the five seeds reach 0.575–0.586 success at $p_{test}=0.1$ and
  0.246–0.257 at $p_{test}=0.2$ (≈98–102% of the DP values, within the ±0.016 standard error), with returns of −307 to −283
  at $p_{test}=0.1$ (DP: −279). They still solve the noise-free task (return 571–573). The learning target includes the
  consequences of slips, so actions whose slips lead into a cliff are valued lower. The slips also make the agent visit, and
  learn to recover from, off-path states.
* $p_{train}=0.2$ brings no further benefit. Its median seed is worse than $p_{train}=0.1$ even when tested at $p=0.2$ (0.23 vs
  0.25), because it finds the goal later (on average at episode 184k vs 151k) and its value estimates are noisier.

**Why one seed per noisy setting fails (snapshot study above).** Seed 1 fails the noise-free task at the end of training for
both $p_{train}=0.1$ and $0.2$, but this is not a failure to learn:
1. Its training success over the last third (0.496 at $p=0.1$, 0.162 at $p=0.2$) is the same as that of the other seeds
   (0.485–0.503 and 0.091–0.156).
2. Most of its earlier snapshots are good: 8 of its 11 snapshots reach ≈0.50–0.62 success at $p_{test}=0.1$.
3. The seeds that look good at the end have bad snapshots too. With $p_{train}=0.1$, 18% of all 55 snapshots fail the
   noise-free task, and every seed has at least one snapshot with ≤ 0.40 success at $p_{test}=0.1$ (seed 0 drops to 0.00 at
   280k episodes). With $p_{train}=0.2$, only 56% of the snapshots solve the noise-free task. Without noise, 100% do.

The cause is the **constant step size** $\alpha=0.1$. Under noise, the TD target of the same $(s,a)$ is random: a slip into a
cliff gives a target near −1000, far below the target after a normal step, so a single update moves $Q(s,a)$ by
$\alpha\cdot 1000 \approx 100$ or more. This can exceed the gap between the best and the second-best action at a critical state. The greedy action at a
critical state therefore keeps flipping, and when it flips to a bad action, the deterministic noise-free rollout gets stuck
until the 100-step time-out (return −679 / −840 instead of ≈570). Noise-free targets are deterministic, so those Q-values
converge and every snapshot behaves identically. Two standard remedies, not applied here so that all $p_{train}$ settings
use the main agent's exact configuration, are a decaying step size (e.g. $\alpha \propto 1/N(s,a)$, which satisfies the
Robbins–Monro conditions) and choosing the final policy by evaluating checkpoints on separate validation episodes.

**Answer.** Action noise makes learning much harder. It needs about 20× the template's budget, the goal is found later, and
the final greedy policy is unstable from one checkpoint to the next. Training with **moderate noise ($p_{train}=0.1$) does
improve robustness**. A converged policy closes almost the whole gap to the optimal noisy policy and loses almost nothing on
the noise-free task. Averaged over *all* 55 snapshots, including the bad ones, it still beats $p_{train}=0$ at $p_{test}=0.1$
(0.512 vs 0.495) and $p_{test}=0.2$ (0.209 vs 0.173), at the price of 18% of snapshots failing the noise-free task. Heavier
noise ($p_{train}=0.2$) brings no additional benefit within the budget (0.564 / 0.322 / 0.124 averaged over its snapshots).