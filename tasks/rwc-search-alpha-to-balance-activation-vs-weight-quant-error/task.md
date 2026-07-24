## Context

SmoothQuant-style activation migration trades quantization difficulty between
activations and weights via a per-channel scale

$$
s_j = \frac{(\max |X_j|)^{\alpha}}{(\max |W_j|)^{1-\alpha}},
$$

where $X_j$ is channel $j$ of the activations and $W_j$ is the matching
output channel of the weights. Migrating with this scale gives

$$
\hat W = W \cdot \mathrm{diag}(s), \qquad \hat X = X \cdot \mathrm{diag}(s)^{-1},
$$

so that the product $\hat X \hat W^\top$ (up to broadcasting) is mathematically
unchanged, but the *dynamic range* is redistributed: a large $\alpha$ shrinks
the activation range (easier to quantize) at the cost of a larger weight
range (harder to quantize), and vice versa. $\alpha=0$ puts all the
difficulty on activations, $\alpha=1$ puts all of it on weights.

There is no closed-form optimum; production libraries grid-search $\alpha$
and pick the value that minimizes the **worse of the two** post-migration
INT8 quantization errors.

## Task

Implement:

```python
def search_best_alpha(W: np.ndarray, X: np.ndarray, alphas: np.ndarray):
    ...
```

* `W` — weight tensor of shape `(C_out, *)`; the first axis indexes output
  channels.
* `X` — activation tensor of shape `(N, C_out, *)`; the batch dimension is
  first, channels second.
* `alphas` — 1-D array of candidate $\alpha$ values in $[0, 1]$.

For every $\alpha_k$ in `alphas`:

1. **Migration scale** (per output channel $j$, matching the formula above):
   $$
   s_j = \frac{(\max |X_j|)^{\alpha_k}}{(\max |W_j|)^{1-\alpha_k}}
   $$
   where $\max|X_j|$ is taken over the batch and all trailing dims of
   channel $j$, and $\max|W_j|$ is taken over all of $W$'s trailing dims of
   channel $j$.

2. **Migrate**: `W_mig = W * s` (broadcast over axis 0), `X_mig = X / s`
   (broadcast over axis 1).

3. **Quantize to INT8**, per tensor, symmetric, round-to-nearest:
   $$
   \text{scale} = \frac{\max|t|}{127}, \qquad
   \hat t = \mathrm{clip}\big(\mathrm{round}(t / \text{scale}), -127, 127\big) \cdot \text{scale}.
   $$
   (If $\max|t| < 10^{-12}$, treat the tensor as already exact — return it
   unchanged.)

4. **Error for this alpha**:
   $$
   \mathrm{err}(\alpha_k) = \max\Big( \mathrm{rel\_err}(X_{\text{mig}}, \hat X_{\text{mig}}),\; \mathrm{rel\_err}(W_{\text{mig}}, \hat W_{\text{mig}}) \Big),
   $$
   where $\mathrm{rel\_err}(a, b) = \lVert b - a \rVert_2 / \lVert a \rVert_2$
   computed over the *flattened* tensor (add a tiny epsilon, $10^{-12}$, to
   the denominator to avoid division by zero).

Return `(best_idx, errors)`:

* `best_idx` — `int`, the index into `alphas` of the $\alpha$ that minimizes
  `err(alpha)` (i.e. `int(np.argmin(errors))`).
* `errors` — 1-D `float64` NumPy array of length `len(alphas)` holding
  `err(alphas[k])` for every `k`.

## Example

```python
import numpy as np

W = np.random.default_rng(0).standard_normal((4, 3))
X = np.random.default_rng(1).standard_normal((10, 4)) * 20.0  # large activations
alphas = np.linspace(0.0, 1.0, 11)

best_idx, errors = search_best_alpha(W, X, alphas)
print(alphas[best_idx], errors[best_idx])
```

Since activations are much larger than weights here, pushing more of
`alpha` toward 1 (migrating range from activations to weights) should lower
the combined error — until the weights themselves start to dominate.

## What the gate checks

The grader builds several `(W, X, alphas)` cases with deliberately imbalanced
activation/weight dynamic ranges (conv-shaped and linear-shaped tensors) and
computes the reference `(idx, errors)` with NumPy, following exactly the
recipe above.

* **`idx_ok`** — your chosen `best_idx` must achieve an error within `1e-6`
  of the true minimum on every case (a tolerant argmin check that only cares
  about picking a genuinely-best alpha, not the exact grid index in case of
  a near-tie).
* **`curve_rel_err`** — the relative L2 error between your full `errors`
  curve and the reference curve, worst-case over all test cases, must be
  `<= 1e-6`. This means your per-alpha error values must match the exact
  recipe above, not just the final argmin.
