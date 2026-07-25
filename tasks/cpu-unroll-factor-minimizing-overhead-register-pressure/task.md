## Context

Unrolling a loop of $N$ iterations by factor $U$ replaces it with
$\lceil N/U \rceil$ "outer" iterations, each doing $U$ original
iterations' worth of work. Two effects trade off against each other:

- **Loop overhead** — the branch and counter update at the bottom of the
  loop — is paid once per *outer* iteration. Modeled as a fixed cost
  $C_{\text{loop}}$, the total is $\lceil N/U \rceil \cdot C_{\text{loop}}$:
  strictly decreasing in $U$.
- **Register pressure** — an unrolled body needs one accumulator per
  in-flight copy. If $U$ exceeds the $R$ registers actually available,
  the extra $U - R$ accumulators spill to the stack, paying a modeled
  $C_{\text{spill}}$ penalty per spilled register, on *every* outer
  iteration.

$$\text{cost}(U) = \left\lceil \frac{N}{U} \right\rceil \cdot \Big(C_{\text{loop}} + \max(0,\, U - R) \cdot C_{\text{spill}}\Big)$$

Pushed too far, unrolling stops paying for itself: the outer-iteration
count is already tiny, so shaving it further barely helps, while every
extra register over the budget adds cost on every remaining iteration.

## Task

Implement both:

```cpp
long unroll_cost(int N, int U, int C_loop, int R, int C_spill);
int  choose_best_unroll(int N, int max_U, int C_loop, int R, int C_spill);
```

`unroll_cost` is the formula above, exactly (integer ceiling division).
`choose_best_unroll` tries every `U` from `1` to `max_U` and returns
whichever minimizes `unroll_cost` (smallest `U` on a tie).

## Example

`N=997, C_loop=60, R=6, C_spill=10`: at `U=6` (exactly at the register
budget, no spill), cost is `167 * 60 = 10020`. At `U=8` (2 spilled
registers), cost drops to `125 * (60+20) = 10000` — the shrinking outer
loop count still wins. At `U=9` (3 spills), cost drops further to
`111 * (60+30) = 9990`, the true minimum; at `U=10` it's already back up
to `100 * (60+40) = 10000` — the spill penalty has caught up.
`choose_best_unroll` returns `9`.

## What the gate checks

`exact_match` on `(U, cost(U))` for two fixed scenarios: `N=1024,
C_loop=20, R=8, C_spill=15` (optimum lands exactly at the register count,
`U=8`), and the `N=997` case above (a prime `N`, so no exact-division
ties — the true optimum, `U=9`, sits strictly past the register budget).
Using `floor` instead of `ceil`, applying the spill penalty even when
`U <= R`, or searching for the *largest* cost instead of the smallest,
returns a different `U` and changes both printed numbers.
