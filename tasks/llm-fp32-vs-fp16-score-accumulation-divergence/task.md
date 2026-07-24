## Context

Scaled dot-product attention computes query-key scores as

$$
S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}} .
$$

The dot product is an accumulation of element-wise products:

$$
Q_iK_j^\top = \sum_{t=1}^{d} Q_{it}K_{jt}.
$$

The accumulation dtype affects the result. In FP16 accumulation, each intermediate addition is rounded to FP16. In FP32 accumulation, the intermediate sum keeps more precision. For large logits, small score differences can change the softmax output.

The attention weights are

$$
A_{ij} = \frac{\exp(S_{ij})}{\sum_k \exp(S_{ik})},
$$

and the final output is

$$
O = AV .
$$

## Task

Implement `attention_fp16_scores(q, k, v)`:

```python
def attention_fp16_scores(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray
) -> np.ndarray:
    ...
```

The inputs are NumPy arrays with shapes $(n,d)$, $(m,d)$, and $(m,h)$. Return the scaled dot-product attention output.

Requirements:

- The query-key dot products must accumulate in FP16.
- Divide scores by $\sqrt{d}$ before applying softmax.
- Use a numerically stable row-wise softmax.
- Return a floating-point NumPy array containing $AV$.

Do not use a default FP32 matrix multiplication for the score calculation.

## Example

```python
import numpy as np

q = np.array([[1, 2]], dtype=np.float32)
k = np.array([[1, 0], [0, 1]], dtype=np.float32)
v = np.array([[3], [5]], dtype=np.float32)

out = attention_fp16_scores(q, k, v)
```

The output is an attention-weighted combination of the value rows. The exact values depend on FP16 score accumulation.

## What the gate checks

The grader computes the reference attention output using an explicit NumPy FP16 accumulation loop and compares the submitted implementation against it.

The error metric is

$$
\max_{i,j}|O_{ij}^{candidate}-O_{ij}^{fp16}|.
$$

The candidate must satisfy

$$
\max_{i,j}|O_{ij}^{candidate}-O_{ij}^{fp16}| \le 10^{-4}.
$$

The grader also computes the same attention operation with FP32 accumulation and checks that the two accumulation modes differ by at least

$$
10^{-3}
$$

on an adversarial tensor, showing that score accumulation precision changes the result.
