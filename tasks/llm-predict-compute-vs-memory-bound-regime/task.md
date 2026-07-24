## Context

In a large language model (LLM) inference pipeline each token is processed by a sequence of linear transformations.  
For a single layer the dominant arithmetic work is a matrix‑vector multiply
$$y = W x + b,$$
where $W \in \mathbb{R}^{d\times d}$, $x,b \in \mathbb{R}^d$.  
The number of floating‑point operations (FLOPs) required for one token is therefore
$$\mathrm{FLOPs}_{\text{token}} = 2\,d^{2},$$
(two multiplications and two additions per matrix entry).

During inference the weight matrix $W$ is reused across all tokens in a sequence, so the total FLOPs for a batch of size $B$ and a sequence length $S$ are
$$\mathrm{FLOPs}_{\text{total}} = 2\,d^{2}\,S.$$

The memory traffic consists of reading the weight matrix once per layer, loading each input vector, and writing each output vector:
$$\mathrm{Bytes}_{\text{token}} = (d^2 + 2d)\,\text{bytes/element},$$
so that the total bytes for a batch are
$$\mathrm{Bytes}_{\text{total}} = (d^2 + 2d)\,\text{bytes/element}.$$

The operational intensity (OI) is defined as FLOPs per byte of memory traffic:
$$\mathrm{OI} = \frac{\mathrm{FLOPs}_{\text{total}}}{\mathrm{Bytes}_{\text{total}}}
= \frac{2\,d^{2}\,S}{(d^2 + 2d)\,\text{bytes/element}}.$$

A hardware roofline model gives a *ridge point* equal to the ratio of peak floating‑point throughput $\mathsf{peak_{flops}}$ and peak memory bandwidth $\mathsf{peak_{bw}}$.  
If $\mathrm{OI}$ exceeds this ridge, the kernel is **compute‑bound**; otherwise it is **memory‑bound**.

## Task

Implement `predict_regime`:

```python
def predict_regime(batch: np.ndarray,
                   seq: np.ndarray,
                   d: np.ndarray,
                   dtype: np.ndarray,
                   peak_flops: np.ndarray,
                   peak_bw: np.ndarray) -> np.ndarray:
    ...
```

The arguments are 1‑D arrays of equal length $N$ describing $N$ independent configurations.  
`dtype` contains string names such as `"float16"` or `"float32"`.  
Return a NumPy array of shape $(N,)$ with integer values: `1` if the configuration is compute‑bound and `0` otherwise.

The implementation must use only NumPy; no explicit Python loops are allowed.  The returned array should be of type `int64`.

## Example

```python
import numpy as np
batch = np.array([8, 16])
seq   = np.array([128, 256])
d     = np.array([1024, 2048])
dtype = np.array(["float32", "float16"])
peak_flops = np.array([200e12, 250e12])   # FLOPs/s
peak_bw    = np.array([1000e9, 1200e9])   # Bytes/s

regime = predict_regime(batch, seq, d, dtype, peak_flops, peak_bw)
print(regime)          # e.g. [1 0]
```

## What the gate checks

The grader computes a reference classification using the exact formulas above and compares it element‑wise with the candidate’s output.  
All entries must match (`exact_match == 1.0`).  No other metrics are evaluated.
