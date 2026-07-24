## Context

LayerNorm normalizes a feature vector $x \in \mathbb{R}^n$ by its own mean and
variance:

$$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i, \qquad
\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \mu)^2, \qquad
y_i = \gamma_i \, \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta_i .$$

The only free choice is *how you compute the variance*. Two textbook single-loop
choices behave very differently in low precision:

- **Two-pass.** First sweep accumulates $S = \sum_i x_i$ to get $\mu = S/n$, then a
  second sweep accumulates $\sum_i (x_i - \mu)^2$. In `float16` the running sum
  $S$ is the weak link: once $S$ grows past a few thousand, the fp16 spacing (ULP)
  becomes larger than the individual terms, so each addition is rounded away and
  $\mu$ drifts. A biased $\mu$ then poisons the second pass.

- **Welford.** A single online sweep keeps a *running mean* and a running sum of
  squared deviations $M_2$:
  $$\mu \mathrel{+}= \frac{x_i - \mu}{i}, \qquad
    M_2 \mathrel{+}= (x_i - \mu_{\text{old}})(x_i - \mu_{\text{new}}), \qquad
    \sigma^2 = M_2/n .$$
  The running mean never grows without bound, so it never falls into the fp16
  saturation regime that wrecks the two-pass sum.

On an ill-conditioned input (a large common offset plus a small spread), the
two paths give visibly different LayerNorm outputs when everything is done in
`float16`. This task asks you to build both and confirm Welford is the more
accurate one.

## Task

Implement two functions, each computing LayerNorm on a 1-D vector **entirely in
`float16`** (cast the inputs, and keep every intermediate — sums, mean,
variance, the normalized output — in `np.float16`):

```python
import numpy as np

def layernorm_fp16_welford(x, gamma, beta, eps=1e-5) -> np.ndarray: ...
def layernorm_fp16_two_pass(x, gamma, beta, eps=1e-5) -> np.ndarray: ...
```

Both take `x`, `gamma`, `beta` as length-$n$ arrays and return a length-$n$
`float16` array. Use the **exact fp16 procedure** below so your result is
reproducible (accumulate strictly left-to-right — do not use a pairwise/`np.sum`
reduction, which rounds differently):

```
x16, g16, b16 = as float16 arrays

# --- two-pass mean/variance ---
s = float16(0)
for xi in x16:            s = float16(s + xi)
mean = float16(s / n)
s2 = float16(0)
for xi in x16:            d = float16(xi - mean); s2 = float16(s2 + float16(d*d))
var = float16(s2 / n)

# --- Welford mean/variance ---
mean = float16(0); M2 = float16(0)
for k, xi in enumerate(x16, start=1):
    delta  = float16(xi - mean)
    mean   = float16(mean + float16(delta / float16(k)))
    delta2 = float16(xi - mean)
    M2     = float16(M2 + float16(delta * delta2))
var = float16(M2 / n)

# --- shared normalization (both paths) ---
inv = float16(1.0 / float16(sqrt(float16(var + float16(eps)))))
y_i = float16(float16(g16_i * float16((x16_i - mean) * inv)) + b16_i)
```

## Example

```python
import numpy as np
x = np.array([100.3, 99.8, 100.1, 100.6], dtype=np.float64)  # offset ~100, tiny spread
g = np.ones(4); b = np.zeros(4)

w = layernorm_fp16_welford(x, g, b)
t = layernorm_fp16_two_pass(x, g, b)
# both return length-4 float16 arrays; on a large-offset vector the two differ,
# and w is markedly closer to the float32 LayerNorm of x than t is.
```

## What the gate checks

The grader builds several ill-conditioned vectors (large offset, small spread,
a few hundred elements) with a fixed seed and, for each, computes an fp32
LayerNorm as ground truth plus its own fp16 Welford and fp16 two-pass
references. Three gates, all against real NumPy oracles:

- $\mathrm{welford\_match\_err} \le 10^{-3}$ — your Welford output matches the
  oracle's fp16 Welford output (you really implemented the specified procedure).
- $\mathrm{two\_pass\_match\_err} \le 10^{-3}$ — same for your two-pass output.
- $\mathrm{err\_ratio} \le 0.5$ — with $e_w$ and $e_t$ the max-abs errors of your
  Welford and two-pass outputs versus the fp32 truth, the worst-case ratio
  $e_w / e_t$ stays at or below $0.5$; i.e. Welford's error is at most half of
  two-pass's on every fixture.
