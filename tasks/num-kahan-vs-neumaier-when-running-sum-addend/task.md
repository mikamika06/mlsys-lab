## Context

Classic **Kahan summation** tracks a running sum $s$ and a lost-bits
compensation $c$. For each new term $x$:

$$
y = x - c, \qquad t = s + y, \qquad c = (t - s) - y, \qquad s = t .
$$

The compensation update $c = (t - s) - y$ implicitly assumes the accumulator
$s$ is the dominant (larger-magnitude) operand of the addition — that is how
it recovers the bits of $y$ that were rounded away when forming $t = s + y$.
That assumption breaks the moment a single addend is **larger** than the
running sum so far: when $|s| < |x|$, it is $s$'s low-order bits that vanish
inside $t$, not $x$'s, so $(t - s) - y$ recovers nothing useful and the lost
bits of $s$ are silently discarded forever.

**Neumaier's modification** (Kahan–Babuška summation) fixes this by checking
which operand is larger before computing the correction:

$$
t = s + x, \qquad
c \mathrel{+}= \begin{cases}
(s - t) + x & \text{if } |s| \ge |x| \\
(x - t) + s & \text{if } |s| < |x|
\end{cases}, \qquad
s = t ,
$$

and returns $s + c$ at the end. This one branch makes the compensation exact
in either regime, at the same $O(1)$ memory and $O(N)$ time cost as Kahan.

## Task

Implement both:

```python
def kahan_sum(x: list[float]) -> float:
    ...

def neumaier_sum(x: list[float]) -> float:
    ...
```

* `x` — a 1-D `float64` list.
* `kahan_sum` must implement the **classic** Kahan recurrence exactly as
  written above (including its blind spot — do not fix it here).
* `neumaier_sum` must implement the magnitude-checked Neumaier recurrence
  above, which stays accurate even when a running sum near zero is followed
  by a much larger addend.
* Both return a Python/Python scalar float, computed with `float64` arithmetic
  throughout (no casting through `float32` or `Decimal`).

## Example

```python

# running sum sits at 1.0, then a huge addend arrives, then it cancels back out
x = [1.0, 1e16, 1.0, 1.0, -1e16, 1.0, 1.0, 1.0]

kahan_sum(x)     # loses several of the unit terms once |s| < |x| — inaccurate
neumaier_sum(x)  # exact: 6.0
```

## What the gate checks

The grader builds several `(small-term, large-scale, -large-scale)` fixtures
with scales from $10^{16}$ to $10^{24}$ — the regime where a running sum near
zero meets a much larger addend — plus the exact worked example above.

1. `neumaier_rel_err` — your `neumaier_sum` compared to `math.fsum` (Python's
   exact, arbitrary-precision-accurate summation) via `rel_err`, worst case
   over all fixtures. Gate: `<= 1e-12`.
2. `kahan_match_err` — your `kahan_sum` compared to an independent reference
   implementation of the *unmodified* classic Kahan recurrence, via
   `max_abs_err`, worst case over all fixtures. This checks you implemented
   Kahan **faithfully** (blind spot included), not that it is accurate. Gate:
   `<= 1e-9`.
3. `neumaier_advantage` — on the worst fixture, `|kahan_sum(x) - fsum(x)| -
   |neumaier_sum(x) - fsum(x)|` computed from **your own** two functions.
   This is the actual head-to-head comparison: Neumaier must measurably beat
   Kahan on the case designed to break it. Gate: `>= 0.1`.
