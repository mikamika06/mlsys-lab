## Context

Symmetric uniform quantizers need a clip value $c$: anything with
$|w| > c$ saturates, anything inside $[-c, c]$ is rounded onto
$2q_{\max}+1$ levels ($q_{\max} = 2^{\text{bits}-1}-1$). The naive
choice $c = \max(|w|)$ protects the single largest outlier in a group
at the cost of a coarse `scale` for everyone else. Methods like PACT
and LSQ instead **learn** the clip value by gradient descent on the
reconstruction error.

Here we learn a single per-group scalar $\alpha$ that scales the
naive range, $c(\alpha) = \alpha \cdot \max(|g|)$, to minimize that
group's quantization MSE:

$$
\text{scale}(\alpha) = \frac{c(\alpha)}{q_{\max}}, \qquad
\hat g_i(\alpha) = \text{scale}(\alpha)\cdot\mathrm{clip}\!\Big(\mathrm{round}\big(\tfrac{g_i}{\text{scale}(\alpha)}\big),\, -q_{\max},\, q_{\max}\Big)
$$

$$
\mathrm{MSE}(\alpha) = \frac{1}{|g|}\sum_i \big(\hat g_i(\alpha) - g_i\big)^2
$$

Because rounding makes $\mathrm{MSE}(\alpha)$ non-smooth, its
derivative is estimated with a **central finite difference** and only
the *sign* of that estimate is used to step $\alpha$ (a sign-gradient
/ RProp-style update — robust to a badly-scaled or noisy gradient
estimate):

$$
g_\alpha \approx \frac{\mathrm{MSE}(\alpha+\varepsilon) - \mathrm{MSE}(\alpha-\varepsilon)}{2\varepsilon}, \qquad
\alpha \leftarrow \mathrm{clip}\big(\alpha - \eta\cdot\mathrm{sign}(g_\alpha),\; 0.2,\; 1.5\big)
$$

starting from $\alpha_0 = 1$ (the naive range).

## Task

Implement `learned_clip_range`:

```python
def learned_clip_range(w: np.ndarray, group_size: int, bits: int,
                        n_steps: int = 25, lr: float = 0.05, eps: float = 1e-3):
    ...
```

- `w`: 1-D `float64` array, `len(w)` a multiple of `group_size`.
- `group_size`: contiguous elements per group (each group gets its own $\alpha$).
- `bits`: quantizer bit width, $q_{\max} = 2^{\text{bits}-1}-1$.
- `n_steps`, `lr` ($\eta$), `eps` ($\varepsilon$): optimizer hyperparameters.

For each contiguous group of `group_size` elements, independently:

1. Start at $\alpha = 1.0$.
2. Repeat `n_steps` times: estimate $g_\alpha$ with the central
   finite difference above, then
   $\alpha \leftarrow \mathrm{clip}(\alpha - \eta\cdot\mathrm{sign}(g_\alpha),\, 0.2,\, 1.5)$.
   (`np.sign(0.0) == 0.0`, so a zero-gradient step leaves $\alpha$
   unchanged.)
3. Record the final $\alpha$ and $\mathrm{MSE}(\alpha)$ for that group.

Return `(alphas, mses)`: two 1-D `float64` arrays of length
`len(w) // group_size`, one entry per group, in group order.

## Example

```python
import numpy as np
w = np.array([1.0, -1.0, 0.05, 9.0])   # one group, group_size=4
alphas, mses = learned_clip_range(w, group_size=4, bits=4, n_steps=25)
# alphas.shape == (1,), mses.shape == (1,)
# alpha drifts below 1.0: shrinking the range (at the cost of clipping
# the outlier 9.0) reduces rounding error for the other three entries
# more than it adds from clipping -- exactly the trade-off a learned
# clip range is meant to find.
```

## What the gate checks

The grader builds several seeded `(w, group_size, bits)` cases and runs
the *exact* algorithm above independently in NumPy (same
finite-difference sign-gradient loop, same hyperparameters, same
`np.sign` convention) to get reference `alphas`/`mses`. Because the
optimizer is deterministic given fixed hyperparameters and a fixed
starting point, a correct implementation reproduces the oracle's
trajectory essentially exactly.

`alpha_rel_err` is the global relative L2 error between your `alphas`
and the oracle's across all cases (must be `<= 1e-4`) — this catches a
wrong clip formula, wrong finite-difference step, or wrong sign
convention. `mse_abs_err` is the worst-case absolute difference between
your final per-group MSE and the oracle's (must be `<= 1e-6`) — this
catches a case where `alphas` happens to look right but the reported
MSE wasn't recomputed at the *final* `alpha`.
