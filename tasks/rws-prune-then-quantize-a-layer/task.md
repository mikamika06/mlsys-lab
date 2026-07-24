## Context

Pruning removes a subset of weights from a neural‑network layer to reduce memory and compute.  
A common pruning pattern is the **2:4 mask**: for every contiguous block of four weights, only two are kept; the other two are set to zero.

After pruning we often quantize the surviving weights so that they can be stored in fewer bits.  In this task we use a signed 4‑bit representation, i.e. values in the range
$$[-8,\dots,7].$$  
The usual per‑group *absmax* scaling is applied: for each block we compute
$$s = \frac{\max_{i\in B} |w_i|}{7},$$
quantize by
$$q_i = \operatorname{clip}\!\bigl(\operatorname{round}(w_i/s),-8,7\bigr),$$
and dequantize back to floating point with
$$\hat w_i = q_i\,s.$$

The final output of the function is the **dequantized** weight matrix; zeros from pruning remain exactly zero.

## Task

Implement `prune_then_quantize(W, group_size)`:

```python
def prune_then_quantize(W: np.ndarray, group_size: int) -> np.ndarray:
    ...
```

* `W` – a 2‑D NumPy array of shape `(n, d)` containing the original weights.  
* `group_size` – the size of each block along the last dimension (e.g. `4` for a 2:4 mask).  

The function must:

1. For every row and every consecutive block of length `group_size`, keep only the two entries with largest absolute value; set all other entries in that block to zero.
2. Quantize the surviving values per block using signed int‑4 scaling as described above, then immediately dequantize back to float64.
3. Return a NumPy array of shape `(n, d)` containing the dequantized weights.

The implementation must use only NumPy operations; no explicit Python loops over individual elements are allowed (loops over rows or blocks are acceptable).

## Example

```python
import numpy as np

W = np.array([[ 0.5, -1.2,  3.4,  0.0],
              [ 2.1,  0.0, -0.7,  4.8]])
# group_size = 4 → one block per row
D_hat = prune_then_quantize(W, 4)
print(D_hat)
```

Possible output (values depend on rounding):

```
[[ 0.5   0.     3.375 0.    ]
 [ 2.0625 0.     -0.75  4.8125]]
```

The zeros correspond to pruned weights; the non‑zero values are the dequantized results of a signed int‑4 quantization.

## What the gate checks

The grader computes an *oracle* implementation that follows the exact algorithm described above and compares your output against it using
$$\text{max_abs_err} = \max_{i,j}\,|\,\hat w_{ij}^{\text{candidate}} - \hat w_{ij}^{\text{oracle}}\!|.$$
Your solution must achieve `max_abs_err <= 1e-5` on all test cases.
