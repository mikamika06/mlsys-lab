## Context

GPTQ-style weight-only quantizers compress a tensor to 4 bits per element
using **per-group asymmetric (affine) quantization**: instead of one
symmetric scale for the whole tensor, the raveled elements are split into
consecutive groups (`group_size` elements each, the last group possibly
shorter), and each group gets its own `(scale, zero)` pair chosen so the
group's exact `[min, max]` range maps onto the full unsigned 4-bit code
range `[0, 15]`:

$$
\text{scale} = \frac{\max(g) - \min(g)}{15}, \qquad
\text{zero} = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{-\min(g)}{\text{scale}}\right),\, 0,\, 15\right).
$$

Each element $x$ in the group is then coded as

$$
\text{code}(x) = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{x}{\text{scale}}\right) + \text{zero},\, 0,\, 15\right),
$$

and dequantized as $\hat x = (\text{code}(x) - \text{zero}) \cdot \text{scale}$.
A constant group ($\max(g) = \min(g)$) has no dynamic range to encode; use
`scale = 1.0` for it so the formulas above stay well-defined.

## Task

Implement `quantize_group_affine_uint4`:

```python
def quantize_group_affine_uint4(W, group_size):
    ...
```

- `W` — a `float` array of any shape.
- `group_size` — int, number of consecutive elements per group when `W` is
  **raveled in row-major (C) order**. The final group may be shorter than
  `group_size` if `W.size` is not a multiple of it.

Return `(codes, scale, zero)`:
- `codes` — `uint8` array, **same shape as `W`**, every entry in `[0, 15]`.
- `scale` — `float64` array of length `ceil(W.size / group_size)`, one
  scale per group, in group order.
- `zero` — `float64` array of the same length, one zero-point per group.

## Example

`W = [0.0, 1.5, -2.5, 7.0, 3.0, -1.0]`, `group_size = 2` gives three
groups: `[0.0, 1.5]`, `[-2.5, 7.0]`, `[3.0, -1.0]`. The second group has
`min=-2.5`, `max=7.0`, so `scale = 9.5/15`, `zero = round(2.5/scale)`, and
each of its two elements is coded independently with that group's
`(scale, zero)`.

## What the gate checks

The grader builds several weight tensors (including one all-zero/constant
one and shapes where `W.size` is not a multiple of `group_size`), computes
the same per-group affine quantization independently, and checks:

1. `codes` match the oracle's codes **exactly** (integer equality) —
   `exact_match`.
2. Dequantizing your `codes` with your own `scale`/`zero` reproduces the
   oracle's dequantized reconstruction to within `1e-8` — `max_abs_err`.

Using one global scale/zero instead of per-group ones, grouping in the
wrong order (e.g. column-major instead of raveled row-major), an off-by-one
in the last partial group, or forgetting the constant-group fallback will
all show up as a large deviation.
