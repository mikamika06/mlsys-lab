## Context

A branch predictor guesses whether a conditional branch will be taken
*before* it's evaluated, so the CPU can keep fetching down the guessed
path instead of stalling. The classic **2-bit saturating counter**
predictor keeps one small state machine per branch:

$$
0 = \text{strongly not-taken} \;\to\; 1 = \text{weakly not-taken} \;\to\; 2 = \text{weakly taken} \;\to\; 3 = \text{strongly taken}
$$

The predictor guesses "taken" iff the state is $\ge 2$. After the real
outcome is known, the counter *saturates* toward it: on a taken branch the
state increments (capped at 3); on a not-taken branch it decrements
(floored at 0). A single not-taken branch inside a mostly-taken loop only
flips the state by one step, not all the way — the "2-bit" part is what
gives it hysteresis a naive 1-bit last-outcome predictor doesn't have.

## Task

Implement

```cpp
int count_mispredicts(const int* branch_ids, const int* outcomes, int n, int num_branches);
```

`branch_ids[i]`/`outcomes[i]` (for `0 <= i < n`) is the `i`-th branch event
in program order: it belongs to branch `branch_ids[i]`
(`0 <= branch_ids[i] < num_branches`), and `outcomes[i]` is `1` if that
branch was actually taken this time, `0` otherwise. Every branch has its
**own independent** 2-bit counter, all starting at state 1 (weakly
not-taken). For each event, in order: predict from the branch's *current*
state, compare to the actual outcome, then update that branch's state.
Return the total number of mispredictions across all `n` events.

## Example

A branch alternating `0, 1, 0, 1, ...` starting from state 1: predict
not-taken (correct on the first `0`), state drops to 0; predict not-taken
again but the actual is `1` — **mispredict**, state rises to 1; predict
not-taken, actual `0` — correct, state drops to 0; predict not-taken,
actual `1` — **mispredict** again. A strictly alternating branch
mispredicts roughly half the time, forever — the classic weak spot of a
per-branch saturating counter.

## What the gate checks

`main.cpp` builds a deterministic, round-robin-interleaved trace across 4
branches with different behaviours (mostly-taken, mostly-not-taken,
strictly alternating, and pseudo-random), calls your function, and prints
the mispredict count. The grader compiles your `.cpp` with the real local
`clang++`, runs it, and requires the printed count to match the
reference's exactly ($\mathrm{exact\_match}=1.0$). Always predicting
"not-taken" (or always "taken") gets some events right by luck but misses
the whole point of tracking state per branch, and lands on the wrong
total.
