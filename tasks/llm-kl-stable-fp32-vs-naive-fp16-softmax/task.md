## Context

Softmax is a ubiquitous activation function in machine learning, mapping an arbitrary vector of logits $z \in \mathbb{R}^m$ to a probability distribution over $m$ classes:

$$\operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{m}\exp(z_j)}.$$

Direct evaluation can suffer from numerical overflow or underflow when the logits contain large magnitude values. A common remedy is to subtract the maximum logit per row before exponentiation:

$$\operatorname{softmax}_{\text{stable}}(z)_i = \frac{\exp(z_i - \max_j z_j)}{\sum_{j=1}^{m}\exp(z_j - \max_k z_k)}.$$

When using reduced precision such as IEEE 754 half‑precision ($\mathrm{float16}$), the loss of dynamic range can exacerbate these issues. In contrast, single‑precision ($\mathrm{float32}$) offers a larger exponent range and typically yields more stable softmax outputs.

The Kullback–Leibler (KL) divergence between two probability distributions $p$ and $q$ over the same support is

$$D_{\text{KL}}(p \,\|\, q)=\sum_{i=1}^{m} p_i \log\!\frac{p_i}{q_i},$$

and its mean over a batch of logits measures how far two softmax implementations diverge on average.

## Task

Implement the function `kl_divergence_fp32_vs_fp16` that takes a 2‑D NumPy array of shape $(n, m)$ containing arbitrary logits and returns the **mean KL divergence** between:

1. The *stable* softmax computed in single precision (`float32`);
2. A *naïve* softmax computed entirely in half precision (`float16`) without any stability tricks.

The function must use only vectorized NumPy operations, no explicit Python loops, and return a `float` (Python scalar) representing the mean KL divergence over all rows.

```python
def kl_divergence_fp32_vs_fp16(logits: np.ndarray) -> float:
    ...
```

## Example

```python
import numpy as np
logits = np.array([[0.0, 1.0, 2.0],
                   [10.0, -5.0, 0.0]])
divergence = kl_divergence_fp32_vs_fp16(logits)
print(divergence)   # e.g., 0.123456789
```

## What the gate checks

The grader generates random batches of logits and computes a reference mean KL divergence using NumPy’s high‑precision (`float64`) arithmetic for both softmax variants. Your implementation is compared against this reference; it must match within a relative error of $10^{-9}$. Any deviation beyond that threshold causes the gate to fail.

The gate metric used in `meta.json` is `"exact_match"`, which will be set to 1.0 if your result passes the tolerance test, otherwise 0.0.
