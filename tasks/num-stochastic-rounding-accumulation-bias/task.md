## Context

This experiment simulates what happens when a small update is repeatedly
added into a *low-precision* accumulator — the classic reason master
weights or optimizer accumulators are kept in higher precision than the
compute path. We model "low precision" directly as a fixed grid: the
accumulator may only take values that are exact multiples of a step `q`, and
after every addition the running total is snapped back onto that grid.

Starting from `start` and adding a constant `c` (with $c < q/2$) `n_steps`
times, the exact mathematical result is

$$
\text{exact} = \text{start} + n_{\text{steps}} \cdot c .
$$

**Round-to-nearest-even (RNE)** snaps every intermediate total to the
*nearest* grid point, deterministically:

$$
a_{i+1} = q \cdot \operatorname{round}\!\left(\frac{a_i + c}{q}\right),
$$

using round-half-to-even for exact ties. Because $c < q/2$, the unrounded
value $a_i + c$ is always closer to $a_i$ than to $a_i + q$ — so **every
single addition rounds straight back to $a_i$**, and the accumulator never
moves at all. This isn't a rare corner case: it's the deterministic, generic
outcome, and it silently discards the entire update every time.

**Stochastic rounding (SR)** instead rounds *probabilistically*, with the
probability of rounding up chosen so the rounding is unbiased in
expectation. If $a_i + c$ lies a fraction $t = (a_i + c - a_i)/q = c/q$ of
the way from $a_i$ up to $a_i + q$, then

$$
a_{i+1} =
\begin{cases}
a_i + q & \text{with probability } t, \\
a_i & \text{with probability } 1-t,
\end{cases}
$$

so $\mathbb{E}[a_{i+1}] = a_i + q\cdot t = a_i + c$: a *single* step is
already unbiased, and the accumulated sum over many steps — or averaged
over independent repetitions — converges to the exact result, unlike RNE
which is stuck at `start` forever.

## Task

Implement two functions:

```python
def accumulate_rne(start: float, c: float, n_steps: int, q: float) -> float:
    ...

def accumulate_stochastic(start: float, c: float, n_steps: int, q: float,
rng: random.Random) -> float:
    ...
```

`accumulate_rne` deterministically applies the RNE snapping rule above
`n_steps` times and returns the final accumulator value.

`accumulate_stochastic` applies the stochastic rounding rule above
`n_steps` times, drawing exactly the randomness it needs from `rng` (e.g.
`rng.random()` per step) so that repeated calls with a fresh, differently
seeded generator give independent trials, and returns the final value.

## Example

```python

start, c, q, n_steps = 1000.0, 0.0003, 0.01, 3000
# exact = 1000.0 + 3000 * 0.0003 = 1000.9

rne = accumulate_rne(start, c, n_steps, q)
# rne == 1000.0  -- stuck; every update rounded away.

rng = random.Random(0)
sr = accumulate_stochastic(start, c, n_steps, q, rng)
# a single stochastic trial lands close to 1000.9, though not exactly on
# it (it's still a random variable) -- averaging over many independent
# trials converges to 1000.9.
```

## What the gate checks

The gate runs two scenarios (different `start`/`c`/`q`/`n_steps`, seeded)
and, for each, computes `exact = start + n_steps * c` directly — a real,
closed-form oracle, since this is just the associativity-free sum of
`n_steps` copies of `c`. It calls your `accumulate_rne` and checks it
matches a from-scratch reference RNE simulation (also derived from the
formula above, not hardcoded). Then, for `K=200` independent
`random.Random(seed)` instances (fixed seeds), it calls your
`accumulate_stochastic` once per seed and averages the `K` results.

The gate metric is the worst case over both scenarios of

$$
\text{rel\_err} = \frac{\left|\,\overline{\text{accumulate\_stochastic}} - \text{exact}\,\right|}{|\text{exact}|},
$$

and requires it to stay below $10^{-4}$ — comfortably tighter than RNE's own
bias in the same setup (RNE misses by roughly $9\times10^{-4}$ relative in
the first scenario, an order of magnitude worse). A solution whose
"stochastic" rounding secretly just does deterministic rounding (ignoring
`rng`, or always rounding down/toward zero) reproduces RNE's bias and fails
the gate; the mean over 200 independent seeds is what makes the check
robust to the run-to-run randomness of a *correct* stochastic implementation
while still catching a non-random one.
