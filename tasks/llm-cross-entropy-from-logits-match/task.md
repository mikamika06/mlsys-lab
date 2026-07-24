## Context

Cross‑entropy is a standard loss for classification problems. For a batch of $N$ samples with logits $\mathbf{z}\in\mathbb{R}^{N\times C}$ and integer targets $\mathbf{t}\in\{0,\dots,C-1\}^N$, the per‑sample cross‑entropy is

$$
\ell_i = -\,\log \frac{\exp(z_{i,t_i})}{\sum_{c=0}^{C-1}\exp(z_{ic})}
      = -\,z_{i,t_i} + \log\!\Bigl(\sum_{c=0}^{C-1}\exp(z_{ic})\Bigr).
$$

The mean loss over the batch is $\frac{1}{N}\sum_{i=1}^N \ell_i$.

Directly computing the softmax with `np.exp` can overflow when logits are large. A numerically stable formulation subtracts the row‑wise maximum before exponentiation:

$$
\log p_{ic}=z_{ic}-m_i-\log\!\Bigl(\sum_{c=0}^{C-1}\exp(z_{ic}-m_i)\Bigr),
\qquad m_i=\max_c z_{ic}.
$$

Using this log‑softmax, the loss becomes

$$
\ell_i = -\,z_{i,t_i}+m_i+\log\!\Bigl(\sum_{c=0}^{C-1}\exp(z_{ic}-m_i)\Bigr).
$$

## Task

Implement `cross_entropy_from_logits(logits, targets)`:

```python
def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    ...
```

`logits` is a 2‑D NumPy array of shape `(N, C)` and `targets` is a 1‑D integer array of length `N`. The function must return the mean cross‑entropy loss as a Python `float`. Use only vectorised NumPy operations; no explicit Python loops. The result should be computed in double precision (`float64`).

## Example

```python
import numpy as np
logits = np.array([[2.0, 1.0, 0.1],
                   [0.5, 2.5, 0.3]])
targets = np.array([0, 1])
loss = cross_entropy_from_logits(logits, targets)
print(loss)   # ≈ 0.6931471805599453
```

## What the gate checks

The grader computes a reference loss with NumPy’s stable log‑softmax and compares your output using the `max_abs_err` scorer from `arena.scorers`. The absolute difference must not exceed $10^{-6}$.
