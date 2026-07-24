## Context

Rotary Position Embedding (RoPE) is a positional encoding scheme that rotates each pair of embedding dimensions by an angle proportional to the token position. For a vector $x \in \mathbb{R}^d$ with even dimension, RoPE defines for each index $i=0,\dots,d/2-1$

$$
\theta_i = \frac{\text{pos}}{10000^{\, 2i/d}},
$$

and applies the rotation

$$
x_{2i}^{(\text{rot})}   = x_{2i}\cos\theta_i - x_{2i+1}\sin\theta_i,\\
x_{2i+1}^{(\text{rot})} = x_{2i}\sin\theta_i + x_{2i+1}\cos\theta_i.
$$

A key property of RoPE is *relative‑position invariance*: the dot product between a query vector $q$ at position $m$ and a key vector $k$ at position $n$ depends only on the difference $\Delta = m-n$. In other words,

$$
\langle \text{RoPE}(q,m),\,\text{RoPE}(k,n)\rangle
= f(q,k,\Delta)
$$

for some function $f$, and not on $m$ or $n$ separately.

## Task

Implement the function `rope_relative_dot` that computes this dot product. The signature is:

```python
def rope_relative_dot(
    q: np.ndarray,
    k: np.ndarray,
    pos_q: int,
    pos_k: int
) -> float:
```

* `q`, `k` are 1‑D NumPy arrays of even length $d$.
* `pos_q`, `pos_k` are integer token positions (can be negative).
* The function must return a Python `float` equal to the dot product of the RoPE‑rotated vectors.

The implementation should use only NumPy operations; no explicit Python loops over dimensions. The result must be computed in double precision (`float64`).

## Example

```python
import numpy as np
q = np.array([1, 0, 0, 1], dtype=np.float64)
k = np.array([0, 1, 1, 0], dtype=np.float64)
# positions 3 and 7
dot = rope_relative_dot(q, k, 3, 7)
print(dot)   # a scalar float
```

## What the gate checks

The grader evaluates your implementation against a NumPy reference for several random test cases. It computes the relative error

$$
\text{rel\_err} = \frac{|\,\hat y - y\,|}{\max(|y|, 10^{-12})},
$$

where $y$ is the reference value and $\hat y$ is your output. The gate requires $\text{rel\_err}\le 1\times10^{-4}$ for all test cases.
