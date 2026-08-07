## Context

fp8 E4M3 has a fixed, small grid of representable magnitudes (max finite
value $448$), so every tensor cast to it needs a **scale** first: divide
by a scale so the tensor's largest magnitude lands near the top of the
grid (using as much of the 8-bit range as possible), quantize, then
multiply back by the scale to dequantize. Using **one scale for the
whole tensor** is wasteful when different attention heads have very
different magnitude ranges — a quiet head gets crushed by a scale sized
for a loud head elsewhere in the same tensor. **Per-head absmax scaling**
gives each head its own scale, sized to that head alone:

$$
\text{scale}_h = \frac{\max_{i,j} |K_{h,i,j}|}{448}, \qquad
\hat{K}_{h,i,j} = \mathrm{round\_to\_e4m3}\!\left(\frac{K_{h,i,j}}{\text{scale}_h}\right) \cdot \text{scale}_h
$$

where $\mathrm{round\_to\_e4m3}$ rounds a value's magnitude to the
**nearest** representable E4M3 grid point (clipped to $448$ if it would
overflow) while preserving sign. If a head is entirely zero, its scale is
undefined by the formula above — treat that head's scale as $1.0$ (its
values are all zero regardless).

## Task

Implement `per_head_absmax_e4m3`:

```python
def per_head_absmax_e4m3(k: list[list[list[float]]]) -> list[list[list[float]]]:
    ...
```

- `k`: a `(heads, seq, head_dim)` float64 array.

For each head independently: compute its absmax scale (or `1.0` if the
head is all zeros), scale the head down, round every value's magnitude to
the nearest representable E4M3 grid point (sign preserved, magnitude
clipped to the format's max finite value of `448`), then scale back up.
Return the dequantized `(heads, seq, head_dim)` array.

## Example

```python

k = [[[0.0] * 4 for _ in range(3)] for _ in range(2)]
k[0] = 100.0    # head 0: uniform magnitude 100
k[1, 0, 0] = 0.01  # head 1: one tiny value, rest zero

out = per_head_absmax_e4m3(k)
# head 0's scale = 100/448 (small); every element quantizes/dequantizes
#   close to 100 (fine detail preserved since the whole head shares one
#   magnitude, right at the top of the grid).
# head 1's scale = 0.01/448 (tiny); the single 0.01 entry also lands
#   right at the top of ITS OWN grid, unaffected by head 0's very
#   different magnitude -- a single shared tensor-wide scale would have
#   crushed head 1's value toward zero.
```

## What the gate checks

The grader builds several `(heads, seq, head_dim)` tensors from a seeded
Python generator — heads with wildly different magnitude scales (some
near-zero, some large), a head that is exactly all zeros, and a mix of
positive/negative values — and computes the reference dequantized tensor
independently in Python: it builds the *real* E4M3 grid from the
sign/exponent/mantissa bit-layout formulas (decoding every representable
code, the same oracle a hardware cast would produce), then reproduces the
per-head absmax-scale-and-round procedure above, never calling your
function.

`rel_err` is the global relative L2 error between your returned tensor
and the oracle's, across every scenario, and the gate requires
`<= 1e-6`. Sharing one scale across all heads instead of per-head,
rounding down/up instead of to the nearest grid point, forgetting to
clip an overflowing value to `448`, or mishandling the all-zero head
(dividing by zero) will all produce a visible mismatch.
