## Context

The cross-entropy loss for a single sample measures how well a model's predicted
distribution $p$ matches a target distribution $q$:

$$\ell(q,\, p) \;=\; -\sum_{k=1}^{K} q_k \,\log p_k$$

where $K$ is the number of classes and $p_k = \exp(z_k) \big/ \sum_{j=1}^{K} \exp(z_j)$
is the softmax of the logits $z \in \mathbb{R}^{K}$.

**Label smoothing** discourages overconfident predictions by mixing the one-hot
target with a uniform distribution.  Given hard target index $y$, the smoothed
target is

$$\tilde{q}_k \;=\; (1-\varepsilon)\,\delta_{k,y} \;+\; \frac{\varepsilon}{K}$$

where $\delta_{k,y} = [k = y]$ and $\varepsilon \in [0, 1]$ is the smoothing
parameter.

A naïve implementation first computes $\text{softmax}(z)$ and then takes $\log$,
which is numerically unstable: when any $z_k$ is large, $\exp(z_k)$ overflows
to `+inf`.  The **fused stable** form pulls out the running maximum:

$$\log p_k \;=\; z_k - m - \log\!\sum_{j=1}^{K}\exp(z_j - m),
\qquad m = \max_{j}\, z_j$$

Because every exponent now satisfies $z_j - m \le 0$, the largest exponential
is $\exp(0) = 1$ and overflow is impossible.  For a batch of $N$ samples the
mean loss is

$$L \;=\; \frac{1}{N}\sum_{n=1}^{N}\;\ell\!\bigl(\tilde{q}^{(n)},\, p^{(n)}\bigr).$$

## Task

Implement `label_smoothed_fused_ce`:

```python
import numpy as np

def label_smoothed_fused_ce(logits, targets, eps=0.1):
    """
    logits  : np.ndarray, shape (N, K), dtype float64 — unnormalized scores
    targets : np.ndarray, shape (N,),    dtype int    — true class indices
    eps     : float — label-smoothing factor in [0, 1]

    Returns: float — mean label-smoothed cross-entropy over the batch
    """
```

Use the fused log-sum-exp trick so that the computation is numerically stable
for arbitrary logit magnitudes.  Return a plain Python `float`.  You may use
only `numpy` (no `scipy`, `torch`, or `jax`).

## Example

```python
import numpy as np

logits  = np.array([[2.0, 1.0, 0.1],
                     [0.5, 2.5, 0.3]])
targets = np.array([0, 1])
loss = label_smoothed_fused_ce(logits, targets, eps=0.1)
# K = 3, eps = 0.1  →  smoothed targets for sample 0: [0.9, 0.05, 0.05]
# loss is a float ≈ 0.58
```

## What the gate checks

The gate runs a self-contained NumPy reference that applies the exact same fused
algorithm and reports **relative error**:

$$\mathrm{rel\_err} = \frac{\bigl|\hat{L} - L^{*}\bigr|}
{\bigl|L^{*}\bigr| + 10^{-12}}$$

The gate requires $\mathrm{rel\_err} < 10^{-10}$ across six test cases:

| Case | $N$ | $K$ | $\varepsilon$ |
|------|-----|-----|---------------|
| 1    | 5   | 10  | 0.1           |
| 2    | 1   | 3   | 0.0           |
| 3    | 1   | 3   | 1.0           |
| 4    | 8   | 100 | 0.2           |
| 5    | 32  | 50  | 0.1           |
| 6    | 16  | 200 | 0.05          |

These cover no-smoothing ($\varepsilon=0$), full smoothing ($\varepsilon=1$),
and a range of batch sizes and class counts.  A correctly vectorized implementation
that builds the smoothed target matrix and applies the fused log-sum-exp will pass.
