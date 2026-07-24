## Context

A branch predictor guesses whether an upcoming conditional branch will be
taken, before the condition is actually evaluated; a wrong guess costs a
pipeline flush. Different predictors trade table size for accuracy:

- **Always-taken**: no state, always guesses taken.
- **1-bit last-outcome**: remembers only the previous outcome and predicts
  it will repeat.
- **2-bit saturating counter**: a single up/down counter in
  $\{0,1,2,3\}$ (predict taken iff $\ge 2$) that only flips its prediction
  after two consecutive wrong guesses in a row — more resistant to a lone
  outlier than the 1-bit scheme.
- **gshare**: a whole *table* of 2-bit saturating counters, indexed by a
  short global history register of the last `hist_bits` actual outcomes,
  so different recent patterns get independently-tracked counters.

## Task

Implement

```cpp
void predictor_mispredicts(const int* outcomes, int n, int hist_bits, int* out);
```

Run all 4 predictors over the same trace `outcomes[0..n)` (`0` =
not-taken, `1` = taken) and write each one's total mispredict count into
`out[0..3]` in this order: always-taken, 1-bit, 2-bit, gshare. Update
rules and initial state are specified exactly in `sol.hpp` — match them
precisely, since even a different initial counter value changes the count.

## Example

For the trace `1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0` (`n=24`)
with `hist_bits=2`: always-taken mispredicts on every `0` (12 of them);
the 1-bit predictor mispredicts on every transition edge (10); the 2-bit
counter, being stickier, needs two consecutive wrong outcomes to flip and
scores 7; gshare's 4-entry table (indices formed from the last 2 outcomes)
scores 9 here — worse than the single 2-bit counter on this particular
short, low-entropy trace, illustrating that more state is not always
better without enough history depth or table capacity to exploit it.

## What the gate checks

`exact_match`: the driver prints all 4 mispredict counts for one fixed 24-
outcome trace. Any wrong update rule, wrong initial counter value, or
wrong history-shift direction changes at least one of the 4 counts and
fails the match; an empty starter leaves the driver's `-1` sentinels in
place.
