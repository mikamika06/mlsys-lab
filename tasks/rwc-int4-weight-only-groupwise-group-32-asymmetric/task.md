## Context

**Weight-only INT4 quantization** (as used by torchao's `Int4WeightOnly`,
and GPTQ-style groupwise quantizers) keeps activations in full precision
and only compresses the weight matrix, storing it as 4-bit codes plus a
small amount of per-group metadata. To keep quantization error low across
a whole row (which can span very different local magnitude ranges), the
row is split into fixed-size **groups** of `group_size` consecutive
input-dim elements, each with its OWN scale and zero-point — this is the
"groupwise" part.

For a group of raw weights $g = \{w_1, \dots, w_{32}\}$, **asymmetric**
min-max quantization to unsigned 4-bit codes ($0$-$15$) is:

$$
\text{scale} = \frac{\max(g) - \min(g)}{15}, \qquad
\text{code}_i = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{w_i - \min(g)}{\text{scale}}\right),\, 0,\, 15\right)
$$

$$
\hat{w}_i = \text{code}_i \cdot \text{scale} + \min(g) .
$$

Unlike symmetric (zero-centered) quantization, this always uses the full
$0$-$15$ code range regardless of whether the group's values are centered
around zero, which is why it is the standard choice for weight-only PTQ.

## Task

Implement `int4_groupwise_asymmetric`:

```python
def int4_groupwise_asymmetric(W: np.ndarray, X: np.ndarray, group_size: int):
    ...
```

* `W` — `(out_features, in_features)` weight matrix, `in_features`
  divisible by `group_size`.
* `X` — `(in_features, batch)` activations (kept full precision).
* `group_size` — number of consecutive input-dim weights sharing one
  `(scale, zero)` pair.

For every row of `W`, split `in_features` into consecutive groups of
`group_size` and quantize each group per the formula above (`zero` is
$\min(g)$). If a group is constant ($\max(g)=\min(g)$), set `scale = 1.0`
and every code to `0` (this reconstructs the constant exactly). Dequantize
the whole weight matrix from `(codes, scales, zeros)` and matmul it with
`X`.

Return `(codes, scales, zeros, output)`:

* `codes` — `(out_features, in_features)` `uint8`, values in `[0, 15]`.
* `scales`, `zeros` — `(out_features, in_features // group_size)` `float64`.
* `output` — `(out_features, batch)` = `dequant(W) @ X`.

## Example

```python
import numpy as np

W = np.array([[0.0, 1.0, 2.0, 3.0]])   # one row, one group (group_size=4)
X = np.eye(4)

codes, scales, zeros, output = int4_groupwise_asymmetric(W, X, group_size=4)
# min=0, max=3 -> scale = 3/15 = 0.2
# codes  = round([0,1,2,3] / 0.2) = [0, 5, 10, 15]
# scales = [[0.2]], zeros = [[0.0]]
# dequant = codes*0.2 + 0 = [0.0, 1.0, 2.0, 3.0]  (exact here since 0.2 divides evenly)
```

## What the gate checks

- **codes_exact** — your `codes` array must exactly match a from-scratch
  min-max reference across 4 random `(W, X)` pairs (`group_size = 32`).
- **max_abs_err** — your `output`, computed from your own dequantized
  weights, must match `dequant(reference W) @ X` to `<= 1e-6` on the same
  cases.

Both gates must pass.
