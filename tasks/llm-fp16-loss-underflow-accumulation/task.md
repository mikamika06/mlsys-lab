## Context

Mixed-precision training keeps activations and logits in `float16`, but the loss
itself must be produced in a wider type. Two different things go wrong when the
whole cross-entropy path stays in binary16.

**1. `exp` overflows.** For a token with logits $z \in \mathbb{R}^{V}$ and target
$y$, the cross-entropy is

$$
\ell = \log\!\Big(\sum_{j=1}^{V} e^{z_j}\Big) - z_y .
$$

`float16` saturates at $65504$, so $e^{z_j}$ becomes `inf` as soon as
$z_j > 11.09$. The standard fix is the max shift, evaluated in `float32`:

$$
\log\sum_j e^{z_j} \;=\; m + \log\sum_j e^{z_j - m},
\qquad m = \max_j z_j .
$$

**2. The accumulator underflows.** A well-fit model produces small per-token
losses, $\ell_t \approx 10^{-2}$. Summing $N \approx 2\cdot10^{4}$ of them
sequentially in `float16` fails long before the end: binary16 has a 10-bit
significand, so once the running sum reaches $s$, any increment smaller than
$\tfrac{1}{2}\,\mathrm{ulp}(s)$ rounds away completely,

$$
\mathrm{fl}_{16}(s + \ell) = s .
$$

The sum *stalls*: every remaining token contributes exactly nothing and the
reported loss freezes at a small fraction of the truth. The same losses
accumulated in `float32` give the right answer. This is why every
mixed-precision trainer keeps the loss accumulator in `float32` even when the
logits are half precision.

## Task

Implement three functions.

```python
import numpy as np

def per_token_ce(logits16: np.ndarray, targets: np.ndarray) -> np.ndarray: ...
def mean_ce_fp32(logits16: np.ndarray, targets: np.ndarray) -> float: ...
def fp16_accum_stall_index(losses: np.ndarray) -> int: ...
```

* `per_token_ce` — `logits16` is a `float16` array of shape $(N, V)$, `targets`
  an integer array of shape $(N,)$. Return a `float32` array of shape $(N,)$
  holding $\ell_t$ for every token. Widen to `float32` *before* exponentiating
  and use the max shift; the grader also feeds logits above $60$, where an
  unshifted or half-precision `exp` produces `inf`.
* `mean_ce_fp32` — return $\frac{1}{N}\sum_t \ell_t$ as a Python `float`,
  accumulated in `float32` or wider.
* `fp16_accum_stall_index` — simulate the naive accumulator on a `float32`
  vector of losses using real binary16 arithmetic, exactly this way:

  ```text
  acc = float16(0)
  for i, x in enumerate(losses):
      l = float16(x)
      new = float16(acc + l)
      if l != 0 and new == acc:   # contribution absorbed
          return i
      acc = new
  return -1                        # never stalled
  ```

  Exact zeros are not a stall. Return a Python `int`.

## Example

```python
import numpy as np

logits16 = np.array([[0.0, 40.0, 1.0]], dtype=np.float16)
targets  = np.array([1])

print(per_token_ce(logits16, targets))   # ~[4.2e-18] - finite, thanks to the max shift
print(mean_ce_fp32(logits16, targets))   # same value as a Python float

losses = np.full(4000, 0.01, dtype=np.float32)
print(fp16_accum_stall_index(losses))
# a few thousand tokens in, the float16 sum has climbed past 32 and
# 0.01 < ulp(32)/2 = 0.0156, so every later token is discarded
```

## What the gate checks

Three gates, all recomputed by the grader from a NumPy oracle on the same
`float16` inputs (nothing is hardcoded):

* **`max_abs_err` ≤ 1e-3** — largest absolute deviation of your `per_token_ce`
  output from a `float64` max-shifted log-sum-exp reference, over two datasets
  (one confident/low-loss, one with logits scaled to roughly $\pm 60$). Any
  `inf` or `nan` fails this gate outright.
* **`mean_loss_abs_err` ≤ 1e-3** — absolute error of `mean_ce_fp32` against the
  `float64` mean. A `float16` accumulator stalls and lands one to two orders of
  magnitude low, so it cannot pass.
* **`stall_exact` == 1.0** — `fp16_accum_stall_index` must match a NumPy
  binary16 simulation on four loss vectors exactly, including one that contains
  exact zeros and one short vector that never stalls (`-1`).
