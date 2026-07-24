## Context

In a transformer, the attention mechanism uses *key* ($K$) and *value* ($V$) tensors that are typically stored in half‑precision floating point (FP16).  
If we store them instead as an 8‑bit integer representation (int8), each element occupies one byte rather than two, yielding a memory saving ratio

$$
\text{ratio} \;=\; \frac{\lVert K_{\text{fp16}},V_{\text{fp16}}\rVert_1}{\lVert K_{\text{fp8}},V_{\text{fp8}}\rVert_1}.
$$

Because the quantised tensors are only an approximation of the FP16 values, the attention output changes.  
For a query matrix $Q$ we compute

$$
A_{\text{fp16}} = \operatorname{softmax}\!\left(\frac{QK_{\text{fp16}}^\top}{\sqrt{d}}\right)V_{\text{fp16}},
$$

and similarly for the dequantised FP8 tensors.  
The *attention output error* is defined as the maximum absolute difference between these two outputs:

$$
E \;=\; \max_{i,j}\bigl|A_{\text{fp16}}(i,j)-A_{\text{fp8}}(i,j)\bigr|.
$$

## Task

Implement a function that returns both the memory‑saving ratio and the attention output error.

```python
def kv_memory_saving_ratio_and_attention_error(
    kv_fp16: np.ndarray,
    kv_fp8:  np.ndarray,
    query:   np.ndarray
) -> tuple[float, float]:
    ...
```

* `kv_fp16` – shape `(2, seq_len, dim)` with dtype `np.float32`.  
  The first slice is the key tensor, the second slice is the value tensor.
* `kv_fp8`  – same shape but dtype `np.uint8`, representing signed integers in
  the range $[-127,\,127]$ that encode values in $[-1,\,1]$ via a linear mapping.
* `query`   – shape `(seq_len, dim)` with dtype `np.float32`.

The function must:

1. Compute  
   $$\text{ratio} = \frac{\text{kv_fp16.nbytes}}{\text{kv_fp8.nbytes}}.$$
2. Dequantise the FP8 tensors by dividing by $127$ to obtain float values in
   $[-1,\,1]$.
3. Compute the attention outputs for both FP16 and dequantised FP8 tensors,
   using the standard scaled‑dot‑product softmax followed by a matrix product
   with the value tensor.
4. Return a tuple `(ratio, error)` where `error` is the maximum absolute
   difference between the two attention outputs.

The implementation must use only NumPy operations; no explicit Python loops are
allowed.

## Example

```python
import numpy as np

# 2‑token sequence, hidden dimension 4
kv_fp16 = np.array([
    [[0.1, -0.3, 0.5, 0.7], [0.2, 0.4, -0.6, -0.8]],
    [[-0.9, 0.1, 0.3, -0.5], [0.7, -0.2, 0.4, 0.6]]
], dtype=np.float32)

# Quantise to int8 in [-127,127]
kv_fp8 = np.round(kv_fp16 * 127).clip(-127, 127).astype(np.uint8)

query = np.array([[0.5, -0.1, 0.3, 0.2]], dtype=np.float32)

ratio, error = kv_memory_saving_ratio_and_attention_error(kv_fp16, kv_fp8, query)
print(ratio)   # 4.0
print(error)   # small value depending on the data
```

## What the gate checks

The grader recomputes the reference ratio and attention‑output error using the
exact algorithm described above.  
Two metrics are reported:

* `size_ratio` – must match the oracle exactly (the grader returns 1.0 if it does, otherwise 0.0).
* `attention_output_error_rel` – the relative difference between the student’s error and the oracle’s error must be ≤ $10^{-9}$.

Both metrics must satisfy their respective gates for a passing solution.
