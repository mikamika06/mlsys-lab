## Context

Computing softmax over a stream you can only see one chunk at a time (the
core trick behind flash attention) means keeping a running max $m$ and a
running unnormalized sum $s = \sum e^{x_i - m}$ WITHOUT ever re-reading
earlier elements. The problem: every term in $s$ was computed relative to
whatever $m$ was *at the time*. The instant a new element's value exceeds
the current $m$, every term already summed into $s$ is relative to the
WRONG (too-small) max and has to be corrected before the new element can
be folded in.

The fix is one multiplication. If the max is about to move from
$m_{\text{old}}$ to $m_{\text{new}} = \max(m_{\text{old}}, x_i)$, every
term already in $s$ needs to be rescaled by

$$\text{factor} = e^{m_{\text{old}} - m_{\text{new}}}$$

so that $s \leftarrow s \cdot \text{factor} + e^{x_i - m_{\text{new}}}$ is
exactly the correct running sum relative to the NEW max. When $x_i$
doesn't exceed the running max, $m_{\text{new}} = m_{\text{old}}$ and the
factor is $e^0 = 1$ — no correction needed, which is exactly the identity
that keeps the online algorithm from needing a special case.

## Task

Write a CUDA-C kernel (single thread — `grid=1, block=1` — the online-max
recurrence is a genuine sequential dependency chain, not something to
parallelize here):

```cpp
__global__ void online_softmax_factors(float* factors, const float* x, int n);
```

`factors[0] = 1.0` (no previous max exists yet). For `i` from `1` to
`n-1`: `new_m = max(m, x[i])`; `factors[i] = expf(m - new_m)`; then update
`m = new_m` before moving to `i+1`.

## Example

On the fixture $x = [2, 1, 5, 3, 5, 6, 0.5, 6]$:

| $i$ | $x_i$ | running $m$ before | $m_{\text{new}}$ | factor |
|---|---|---|---|---|
| 0 | 2   | —   | 2 | $1.0$ (base case) |
| 1 | 1   | 2   | 2 | $e^{0} = 1.0$ (max unchanged) |
| 2 | 5   | 2   | 5 | $e^{-3} \approx 0.049787$ |
| 3 | 3   | 5   | 5 | $1.0$ |
| 4 | 5   | 5   | 5 | $1.0$ (tie — still unchanged) |
| 5 | 6   | 5   | 6 | $e^{-1} \approx 0.367879$ |
| 6 | 0.5 | 6   | 6 | $1.0$ |
| 7 | 6   | 6   | 6 | $1.0$ (tie) |

```
factors = [1.0, 1.0, 0.04978707, 1.0, 1.0, 0.36787944, 1.0, 1.0]
```

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it on the
software GPU over the fixed 8-value fixture, requiring `max_abs_err <=
1e-6` against the numpy running-max oracle. Deriving the factor from the
NEW max relative to the OLD one instead of the reverse (`expf(new_m - m)`,
which is $\geq 1$ instead of $\leq 1$ whenever the max updates) gets 6 of
the 8 entries right (everywhere the max doesn't move, both directions give
$1.0$) but is wrong by roughly $2900\times$ and $2.4\times$ on the two
update steps — the gate's tolerance is far too tight for that to slip
through. The empty starter leaves every entry at its `-1.0` sentinel.
