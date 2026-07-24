## Context

Optimization-based weight quantizers such as **AutoRound** improve on plain
round-to-nearest (RTN) by learning a small per-weight rounding perturbation
$V$ that is added *before* rounding, so that some borderline weights get
pushed to round up instead of down (or vice-versa) — whichever reduces the
group's reconstruction error most. For a per-group asymmetric affine 4-bit
quantizer with group scale $s$ and zero-point $z$, the perturbed code is

$$
\text{code}(x) = \operatorname{clip}\!\Big(\operatorname{round}\!\big(\tfrac{x}{s} + V\big) + z,\; 0,\; 15\Big).
$$

**This task is the $V=0$ baseline**: with no learned perturbation, the
formula collapses to plain round-to-nearest,

$$
\text{code}(x) = \operatorname{clip}\!\Big(\operatorname{round}\!\big(\tfrac{x}{s}\big) + z,\; 0,\; 15\Big).
$$

It is what AutoRound (and every other optimization-based quantizer) starts
from before any tuning happens, and it is the standard GPTQ/RTN reference
point every improvement is measured against.

The tensor is split into consecutive groups of `group_size` elements (the
raveled array, row-major order; the last group may be shorter). Each
group gets its own $(s, z)$ chosen so the group's exact `[min, max]` maps
onto the unsigned 4-bit code range `[0, 15]`:

$$
s = \frac{\max(g) - \min(g)}{15}, \qquad
z = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{-\min(g)}{s}\right),\, 0,\, 15\right).
$$

A constant group ($\max(g)=\min(g)$) has no range to encode; use $s=1.0$ for
it so the formulas stay well-defined. Dequantization is
$\hat x = (\text{code}(x) - z)\cdot s$.

## Task

Implement `quantize_dequant_rtn_v0`:

```python
def quantize_dequant_rtn_v0(W: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    ...
```

- `W` — a `float` array of any shape.
- `group_size` — number of consecutive elements per group when `W` is
  **raveled in row-major (C) order**.

Return `(codes, W_dq)`:
- `codes` — `uint8` array, **same shape as `W`**, every entry in `[0, 15]`,
  the round-to-nearest ($V=0$) 4-bit code of each element under its group's
  `(s, z)`.
- `W_dq` — `float64` array, **same shape as `W`**, the dequantized
  reconstruction $\hat x = (\text{code} - z)\cdot s$.

## Example

`W = [0.0, 1.5, -2.5, 7.0]`, `group_size = 4` is a single group with
`min=-2.5`, `max=7.0`, so `s = 9.5/15`. Each element's code is
`round(x/s) + z`, clipped to `[0, 15]`, and `W_dq` reconstructs each code
back with the same `(s, z)`.

## What the gate checks

The grader loads a fixture weight tensor (`ar_w.npy`, shaped like a real
linear-layer weight, quantized with `group_size=32`) plus several synthetic
tensors (including a constant group and a size not divisible by
`group_size`), computes the same per-group $V=0$ RTN quantization
independently, and checks:

1. `codes` match the oracle's codes **exactly** — `exact_match`.
2. `W_dq` matches the oracle's reconstruction to within $10^{-6}$ —
   `max_abs_err`.

Using one global scale instead of per-group ones, grouping in the wrong
order, an off-by-one on the last partial group, forgetting the
constant-group fallback, or rounding half-away-from-zero instead of banker's
`np.rint`-style rounding will all show up as a deviation from the oracle.
