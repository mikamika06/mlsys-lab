## Context

Quantization is the process of mapping a continuous range of values to a discrete set, typically for efficient storage or computation on hardware with limited precision.  
A *symmetric* quantizer uses a single scale factor and no zero‑point offset; it maps the real value $x$ to an integer $q$ via

$$
q = \operatorname{round}\!\left(\frac{x}{s}\right),\qquad s>0,
$$

and reconstructs with

$$
\hat x = q\,s .
$$

For 8‑bit signed integers the representable range is $\{-128,\dots,127\}$.  
To avoid bias we choose a scale that maps the maximum absolute value of the tensor to $127$:

$$
s = \frac{\max_i |W_i|}{127}.
$$

If all entries are zero we set $s=1$ to keep the formula well‑defined.

## Task

Implement `per_tensor_int8_round_trip`:

```python
def per_tensor_int8_round_trip(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...
```

The function receives a NumPy array `W` of arbitrary shape and returns a pair `(q, dq)` where

* `q` is the quantized tensor as a signed 8‑bit integer array (`dtype=np.int8`);
* `dq` is the dequantized tensor in double precision (`dtype=np.float64`).

The algorithm must follow the symmetric scheme described above.  
Do **not** use any external libraries beyond NumPy.

## Example

```python
import numpy as np
W = np.array([[0, 1], [-2, 3]])
q, dq = per_tensor_int8_round_trip(W)
print(q)   # [[  0   4]
           # [ -5  12]]
print(dq)  # [[0.        0.03149606]
           # [-0.06299213 0.09448819]]
```

The dequantized values are obtained by multiplying the integer codes by the computed scale.

## What the gate checks

A single metric is evaluated:

* **Relative error** –  
  $$\mathrm{rel\_err} = \frac{\lVert W - \hat W\rVert_2}{\lVert W\rVert_2 + 10^{-12}}$$  

  The candidate passes if `rel_err <= 0.02`.  
  This ensures that the round‑trip faithfully reproduces the original tensor within a few percent.

The grader generates several random tensors, runs your implementation, and compares the dequantized result to the ground truth computed by the reference solution. Any deviation beyond the threshold causes the gate to fail.

---
