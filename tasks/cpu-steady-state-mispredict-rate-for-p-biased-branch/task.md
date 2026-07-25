## Context

A 2-bit saturating counter predicts TAKEN whenever its state $s \in
\{0,1,2,3\}$ is $\ge 2$, and after the real outcome it moves one step
toward whichever direction the outcome favors, saturating at the ends:
$s \to \min(s+1, 3)$ on TAKEN, $s \to \max(s-1, 0)$ on NOT-TAKEN.

For a branch that is TAKEN independently with fixed probability $p$
every single time (no exploitable pattern, just a bias), the counter's
state is a birth-death Markov chain: from state $i$, it moves to $i+1$
with probability $p$ and to $i-1$ with probability $q = 1-p$ (the
boundary states self-loop on the direction that would go out of range).
Its stationary distribution $\pi$ satisfies detailed balance,
$\pi_i \, p = \pi_{i+1} \, q$, so with $r = p/q$:

$$
\pi_i = \frac{r^i}{1 + r + r^2 + r^3}, \qquad i = 0,1,2,3.
$$

The steady-state mispredict rate is the chance of guessing NOT-TAKEN
($s<2$) while the branch is actually TAKEN, plus the chance of guessing
TAKEN ($s \ge 2$) while it's actually NOT-TAKEN:

$$
p(\pi_0+\pi_1) + q(\pi_2+\pi_3)
$$

Substituting $\pi$ and simplifying (using $1+r+r^2+r^3 =
(1+r)(1+r^2)$) collapses this all the way down to

$$
\boxed{\text{mispredict\_rate}(p) = \dfrac{p\,(1-p)}{p^2 + (1-p)^2}}
$$

Notice this is *not* the same as $\min(p, 1-p)$, the rate a trivial
"always predict the majority outcome" rule would get: at $p=0.3$ the
majority rule mispredicts $30\%$ of the time, but the 2-bit counter
actually mispredicts about $36.2\%$ of the time. Hysteresis that helps
on *patterned* branches costs extra mispredictions on a purely random
one -- every trip through a weak state on the "wrong side" after a
streak against the bias is a misprediction the majority rule wouldn't
have made.

## Task

Implement:

```cpp
double steady_state_mispredict_rate(double p);
```

Return $\dfrac{p(1-p)}{p^2+(1-p)^2}$.

## Example

For $p = 0.7$: $q = 0.3$, so the rate is
$\dfrac{0.7 \times 0.3}{0.7^2+0.3^2} = \dfrac{0.21}{0.58} \approx
0.362069$ -- noticeably above the naive $\min(0.7,0.3)=0.3$ a majority
predictor would achieve.

## What the gate checks

`main.cpp` calls `steady_state_mispredict_rate` at 5 fixed bias values
($p \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$) and, for each, also runs an
independent 200,000-sample deterministic simulation (seeded `xorshift32`
PRNG, no wall-clock, no library `rand()`) of the actual 2-bit counter
against an i.i.d. Bernoulli($p$) trace, printing the candidate's
theoretical rate next to that empirical one. In the reference's own
output the two always agree to within about `0.002` -- normal sampling
noise at this trace length -- which is itself a live check that the
closed form is right. The candidate's full stdout is compared
byte-for-byte (`exact_match = 1.0`) against the reference's. The
plausible-but-wrong $\min(p, 1-p)$ formula prints the right value only
at $p=0.5$ (where the two coincide) and is off by `6` percentage points
at $p \in \{0.3, 0.7\}$, visibly diverging from that case's own
simulated rate too.
