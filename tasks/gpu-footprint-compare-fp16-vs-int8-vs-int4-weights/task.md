## Context

Modern deep‑learning workloads on GPUs store weight tensors in GPU memory, and the choice of numeric format directly impacts both **memory footprint** and **throughput**.  
A half‑precision ($16$‑bit) floating point value occupies $2$ bytes per element, whereas an $8$‑bit signed integer uses only $1$ byte, and a packed $4$‑bit code can store two values in a single byte.

When we quantise a tensor $W \in \mathbb{R}^{n\times d}$ into lower‑bit representations we typically split the storage into

* **codes** – the quantised integer values per element,
* **scales** – per‑channel or per‑matrix multipliers that recover approximate magnitudes.

The total footprint of a quantised representation is thus  

$$
\lVert C \rVert_{\text{bytes}} + \lVert S \rVert_{\text{bytes}},
$$

where $C$ denotes all code arrays and $S$ all scale arrays.  
A useful metric for comparing two formats is the **size ratio**

$$
\text{size\_ratio}(W, C, S) = 
\frac{\text{bytes of } W}
     {\sum_i \text{bytes of each part in }\{C,S\}} .
$$

For a fair comparison the *reference* $W$ should be kept in its original FP16 form; we then evaluate how many times larger or smaller a quantised representation is relative to that baseline.

## Task

Implement the function  

```python
def size_ratio_fp16_quantized(
    fp16_weights: np.ndarray,
    int8_codes: np.ndarray,
    int8_scales: np.ndarray,
    int4_codes: np.ndarray,
    int4_scales: np.ndarray
) -> Tuple[float, float]:
```

* `fp16_weights` – a $2$‑byte per element NumPy array (`dtype=np.float16`) of shape $(n,d)$.
* `int8_codes` and `int8_scales` – NumPy arrays that together store an INT8 representation.  
  The **codes** have the same spatial shape as `fp16_weights` and use a signed byte type.  
  The **scales** are typically one per column (or channel) and can be any numeric dtype.
* `int4_codes` – a packed $4$‑bit representation stored in an unsigned byte array.  
  Two codes fit into each element; the shape should thus have half as many columns as `fp16_weights`.  
  For example, if the weight tensor has shape $(n,d)$, then `int4_codes` might have shape $(n,\lceil d/2\rceil)`.
* `int4_scales` – a scale array analogous to `int8_scales`.

The function must compute and return two floating‑point values:

1. The size ratio of the **INT8** representation with respect to the FP16 baseline.
2. The size ratio of the **INT4** representation with respect to the FP16 baseline.

Both results should be expressed as `float` (Python built‑in) and rounded to machine precision only by the underlying NumPy operations; no additional rounding is required.

## Example

```python
import numpy as np

# 64×128 weight tensor in FP16
W_fp16 = np.random.randn(64, 128).astype(np.float16)

# INT8 representation (codes + per‑column scales)
int8_codes   = W_fp16.astype(np.int8)                 # same shape, signed byte
int8_scales  = np.abs(np.random.rand(128)).astype(np.float32)  # one scale per column

# INT4 representation (packed codes + per‑column scales)
int4_code_bytes = np.random.randint(
    0, 256, size=(64, 128//2), dtype=np.uint8
)
int4_codes   = int4_code_bytes                         # each byte holds two 4‑bit values
int4_scales  = np.abs(np.random.rand(128)).astype(np.float32)

r_int8, r_int4 = size_ratio_fp16_quantized(
    W_fp16,
    int8_codes, int8_scales,
    int4_codes, int4_scales
)

print(r_int8)   # ≈ 1.9  (fp16 footprint is about twice the INT8 size)
print(r_int4)   # ≈ 3.5  (fp16 footprint is about three‑and‑a‑half times the INT4 size)
```

## What the gate checks

Two gates evaluate the returned ratios:

* `int8_size_ratio >= 0.5` – verifies that the INT8 representation is at most twice as large as the FP16 baseline.
* `int4_size_ratio >= 0.25` – verifies that the packed INT4 representation is at most four times as large as the FP16 baseline.

Both values are computed by the grader against a reference implementation that uses the same NumPy operations, so any incorrect arithmetic or mistaken dtype handling will be caught.
