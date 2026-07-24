## Context

In speculative decoding, a cheap draft model proposes $k$ tokens per block
and a target model verifies them in one forward pass. If the per-step
acceptance probability is $\alpha \in [0, 1]$, the expected number of
target tokens produced by one verification cycle is the geometric sum

$$
E(\alpha, k) = \sum_{i=0}^{k} \alpha^{i} = 1 + \alpha + \alpha^2 + \dots + \alpha^k .
$$

Let $c$ be the draft model's cost relative to one target-model forward
pass ($0 \le c < 1$, since a draft model that costs as much as the target
model defeats the purpose). One verification cycle costs $1 + kc$ relative
units (one target pass plus $k$ draft passes), so the modeled speedup over
plain token-by-token target-only decoding is

$$
S(\alpha, k, c) = \frac{E(\alpha, k)}{1 + kc}.
$$

$S$ is strictly increasing in $\alpha$ (each extra accepted token is
"free" verification, adding to $E$ without adding cost), running from
$S(0,k,c) = \frac{1}{1+kc} \le 1$ up to $S(1,k,c) = \frac{k+1}{1+kc} > 1$
(since $c < 1$). Somewhere in between there is exactly one **break-even**
acceptance rate $\alpha^\star$ where speculative decoding is neither faster
nor slower than plain decoding:

$$
E(\alpha^\star, k) = 1 + kc \quad\Longleftrightarrow\quad S(\alpha^\star, k, c) = 1.
$$

Below $\alpha^\star$, the draft model doesn't get accepted often enough to
pay for its own extra forward passes; above it, speculation is a net win.

## Task

Implement `break_even_alpha(configs)`:

```python
def break_even_alpha(configs: np.ndarray) -> np.ndarray:
    ...
```

`configs` is a NumPy array of shape $(n, 2)$. Each row is `[c, k]`:

- `c`: the draft-to-target cost ratio, $0 \le c < 1$.
- `k`: the draft block length (a non-negative integer, given as a float).

For every row, solve

$$
1 + \alpha^\star + (\alpha^\star)^2 + \dots + (\alpha^\star)^{k} = 1 + kc
$$

for the unique root $\alpha^\star \in [0, 1]$ (there is always exactly one,
since the left-hand side is continuous and strictly increasing in
$\alpha^\star$ from $1$ at $\alpha^\star=0$ to $k+1$ at $\alpha^\star=1$,
and $1 \le 1+kc \le k+1$ whenever $0 \le c < 1$). There is no closed form
for general $k$ — solve numerically (e.g. bisection on $\alpha \in [0,1]$,
since the left-hand side is monotonic).

Return a NumPy array of shape $(n,)$ with $\alpha^\star$ for each row.

## Example

```python
import numpy as np

configs = np.array([
    [0.2, 2],   # c=0.2, k=2
])

alpha_star = break_even_alpha(configs)
# solves 1 + a + a^2 == 1 + 2*0.2 == 1.4, i.e. a + a^2 == 0.4
# alpha_star[0] ~= 0.306226...
```

## What the gate checks

The gate builds several `(c, k)` configurations with $0 \le c < 1$ and
solves the same break-even equation itself, numerically, via bisection on
$\alpha \in [0,1]$ (200 iterations, well past `float64` precision). It
includes the boundary case $c=0$ (where $\alpha^\star = 0$ exactly, since
even zero acceptance already breaks even against a free draft) and several
$k$ values. Your returned $\alpha^\star$ values are compared against this
oracle with relative error; the error must stay below $10^{-6}$ on every
case. Confusing the direction of the inequality (solving for where
speculation becomes strictly worse instead of the equality point), or
using a fixed, too-coarse iteration count / tolerance for large $k$, are
the usual ways to miss the threshold.
