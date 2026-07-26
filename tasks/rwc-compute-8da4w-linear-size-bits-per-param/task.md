## Context

In many quantised linear layers the weights are stored as 4‑bit integers (int4) and each group of columns shares a single floating‑point scale.  
For an output dimension \(O\) and input dimension \(I\), let \(\mathbf{W}\in\mathbb{R}^{O\times I}\) be the full‑precision weight matrix.  The quantised representation consists of

* **Packed int4 weights** – two weights share one byte.
* **Per‑group FP16 scales** – a single 16‑bit value per group of \(g\) input columns.

The *effective bits per parameter* is defined as the total number of bits used to store the quantised layer divided by the number of original parameters \(|O\times I|\).  
The *size ratio* compares this storage against the naïve FP16 representation (2 bytes per weight).

## Task

Implement `compute_8da4w_efficiency`:

```python
def compute_8da4w_efficiency(weight_shape: tuple[int, int], group_size: int) -> tuple[float, float]:
    ...
```

* `weight_shape` is a 2‑tuple `(out_features, in_features)`.
* `group_size` is the number of input columns that share one FP16 scale.
* Return a tuple `(bits_per_param, size_ratio)` where:
  * `bits_per_param` is the total bits used divided by the number of weights.
  * `size_ratio` is \(\frac{2\,|O\times I|}{\text{total bytes}}\).

The function must use only integer arithmetic and return floats.

## Example

```python
>>> compute_8da4w_efficiency((128, 256), 32)
(4.00390625, 3.9960975609756098)
```

Explanation:  
* 128 × 256 = 32768 weights → 16384 bytes of packed int4 (2 weights per byte).  
* 256/32 = 8 groups → 16 bytes of FP16 scales.  
* Total = 16400 bytes = 131200 bits, so \(131200 / 32768 = 4.00390625\) bits per param — the 0.0039 above the nominal 4 is what the scales cost.  
* Size ratio against fp16 = \((32768 \times 2)/16400 = 3.9960975609756098\).  

(Your implementation should produce the exact numbers.)

## What the gate checks

Two metrics are evaluated:

1. `bits_per_param_rel_err` – relative error between your result and a NumPy oracle.
2. `size_ratio_rel_err` – relative error of the size ratio against the same oracle.

Both must be ≤ \(10^{-9}\).  The grader recomputes the reference values on the fly; no hard‑coded numbers are used.
