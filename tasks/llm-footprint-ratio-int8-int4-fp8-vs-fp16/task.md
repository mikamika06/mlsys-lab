## Context

Memory footprint of language‑model parameters is a key cost factor. A common optimisation is to quantise the weight tensors from 16‑bit floating point (FP16) down to lower‑precision integer or reduced‑precision floating formats such as INT8, INT4 or FP8. The total size of a quantised model consists of two parts:

1. **Quantised weights** – each element stored in the chosen numeric format.
2. **Scale factors** – one per tensor, channel or group depending on the granularity.

For a weight matrix $W \in \mathbb{R}^{n\times d}$ the baseline size in FP16 is
$$\lvert W\rvert_{\text{FP16}} = 2\, n\,d \;\text{bytes}.$$

If we quantise to INT8 per‑tensor, each element occupies one byte and a single $32$‑bit scale factor must be stored:
$$\lvert W\rvert_{\text{INT8/pt}} = 1\,n\,d + 4 \;\text{bytes}.$$

For INT4 per‑channel we pack two elements into one byte, so the weight size is
$$\lvert W\rvert_{\text{INT4/pc}} = \bigl\lceil n/2\bigr\rceil d + 4d \;\text{bytes},$$
where $4d$ bytes are for a $32$‑bit scale per channel.

The **footprint ratio** is defined as the total size of the quantised representation divided by the FP16 baseline:
$$R = \frac{\lvert W\rvert_{\text{quant}}}{\lvert W\rvert_{\text{FP16}}}.$$

A ratio $R<1$ indicates a compression benefit.

## Task

Implement `compute_footprint_ratio(weights, scheme="int8", granularity="per_tensor")`:

```python
def compute_footprint_ratio(
    weights: np.ndarray,
    scheme: str = "int8",
    granularity: str = "per_tensor"
) -> float:
    ...
```

* `weights` – a 2‑D NumPy array of shape $(n,d)$ containing the original FP16 parameters.
* `scheme` – `"int8"` or `"int4"`.
* `granularity` – `"per_tensor"` for INT8, `"per_channel"` for INT4.

The function must return the footprint ratio $R$ as a Python float. It should use only NumPy operations and no explicit loops over elements.

## Example

```python
import numpy as np
W = np.random.randn(10, 8).astype(np.float16)
ratio = compute_footprint_ratio(W, scheme="int8", granularity="per_tensor")
# ratio ≈ (1*80 + 4) / (2*80) = 84/160 = 0.525
```

## What the gate checks

The grader computes a reference ratio using NumPy’s dtype sizes and compares it to your output with a tolerance of $10^{-9}$. The function must handle both supported schemes and granularity combinations correctly; any deviation causes the gate to fail.
