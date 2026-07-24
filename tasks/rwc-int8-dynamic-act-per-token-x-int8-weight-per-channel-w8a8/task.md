## Context

In many quantized neural‑network backends the activations of a layer are first
scaled **per token** (row) and then cast to an 8‑bit signed integer.  
Similarly, each output channel of the weight matrix is scaled **per column**
and stored as int8.  
The forward pass therefore consists of three stages:

1. **Quantisation**  
   For a row $i$ of activations $A_{i,:}$ we compute a scale
   $$s_i = \frac{\max_j |A_{ij}|}{127}$$
   and store the integer representation
   $$a^{\text{int}}_{ij}= \operatorname{round}\!\left(\frac{A_{ij}}{s_i}\right),$$
   clipped to $[-128,127]$.  
   For a column $k$ of weights $W_{:,k}$ we compute
   $$t_k = \frac{\max_i |W_{ik}|}{127}$$
   and store
   $$w^{\text{int}}_{ik}= \operatorname{round}\!\left(\frac{W_{ik}}{t_k}\right).$$

2. **Integer matrix multiplication**  
   The product is accumulated in 32‑bit integers:
   $$Y^{\text{int}} = a^{\text{int}}\; W^{\text{int}}.$$

3. **Dequantisation**  
   The final floating point result is obtained by multiplying the integer
   product with the outer product of the two scale vectors:
   $$Y_{ik}= Y^{\text{int}}_{ik}\;\bigl(s_i\,t_k\bigr).$$

This scheme, known as **W8A8** in some libraries, keeps the arithmetic fast
while preserving a good approximation to the full‑precision result.

## Task

Implement the function

```python
def int8_dynamic_act_per_token_x_int8_weight_per_channel(A: np.ndarray,
                                                         W: np.ndarray) -> np.ndarray:
    ...
```

* `A` is an `(n, d)` array of activations (dtype float64).  
* `W` is a `(d, m)` weight matrix (dtype float64).  
* The function must perform the three stages described above and return
  an `(n, m)` array of dtype float64.

The implementation should be fully vectorised; no explicit Python loops over
rows or columns are allowed.  Handle zero‑max rows/columns by setting the
corresponding scale to `1.0` (so that all quantised values become zero).

## Example

```python
import numpy as np
A = np.array([[0, 2], [3, -4]], dtype=np.float64)
W = np.array([[5, -6], [7, 8]], dtype=np.float64)

Y = int8_dynamic_act_per_token_x_int8_weight_per_channel(A, W)
print(Y)
```

The output will be close to the full‑precision product `A @ W` but with a
small quantisation error.

## What the gate checks

The grader computes a reference result using exactly the same algorithm
described above.  It then evaluates the **maximum absolute error**

$$\max_{i,k} |\,Y^{\text{ref}}_{ik}-Y^{\text{sol}}_{ik}\,|$$

and requires this value to be at most $10^{-4}$.

The reference is computed on three random test cases; no hard‑coded
expected values are used.  A correct implementation will therefore pass the
gate automatically, while any deviation (e.g., missing weight scaling or
incorrect clipping) will cause a failure.
