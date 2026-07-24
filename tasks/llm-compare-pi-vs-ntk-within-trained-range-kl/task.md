## Context

In transformer models the attention mechanism produces a probability distribution over tokens for each query position.  
Given two sets of logits $z^{(1)}$ and $z^{(2)}$, we can compare their induced distributions by the Kullback–Leibler divergence

$$D_{\mathrm{KL}}\!\bigl(p^{(1)} \,\|\, p^{(2)}\bigr)
   = \sum_{i} p_i^{(1)} \log \frac{p_i^{(1)}}{p_i^{(2)}},$$

where $p^{(k)}=\operatorname{softmax}(z^{(k)})$.  
The mean KL over all query positions is a convenient scalar metric for comparing two attention variants.

In this task we compare the *Position Interpolation* (PI) and *NTK‑aware* extensions of a base attention model.  For each method we compute its mean KL divergence relative to the base logits.

## Task

Implement `compare_pi_ntk(base_logits, pi_logits, ntk_logits)`:

```python
def compare_pi_ntk(base_logits: np.ndarray,
                   pi_logits: np.ndarray,
                   ntk_logits: np.ndarray) -> tuple[float, float]:
    ...
```

* `base_logits`, `pi_logits`, and `ntk_logits` are 2‑D NumPy arrays of shape `(n, m)` where each row contains the raw attention logits for a query position.
* The function must return a tuple `(pi_kl, ntk_kl)` where
  * `pi_kl` is the mean KL divergence between `pi_logits` and `base_logits`,
  * `ntk_kl` is the mean KL divergence between `ntk_logits` and `base_logits`.
* Use only NumPy; no explicit Python loops over rows.

## Example

```python
import numpy as np
from compare_pi_ntk import compare_pi_ntk

# Random logits for illustration
rng = np.random.default_rng(42)
base = rng.standard_normal((3, 4))
pi   = rng.standard_normal((3, 4))
ntk  = rng.standard_normal((3, 4))

pi_kl, ntk_kl = compare_pi_ntk(base, pi, ntk)
print(pi_kl)   # e.g. 0.312
print(ntk_kl)  # e.g. 0.287
```

The exact numbers depend on the random seed; the important part is that both values are finite floats.

## What the gate checks

* **Correctness** – The grader computes the reference mean KL divergences using `arena.scorers.mean_kl` and compares them to your output.  Your implementation must match within a relative tolerance of $10^{-9}$.
* **Performance** – No explicit Python loops over rows are allowed; the solution should be fully vectorised.

The gate will pass only if both returned KL values equal the reference values (within tolerance).
