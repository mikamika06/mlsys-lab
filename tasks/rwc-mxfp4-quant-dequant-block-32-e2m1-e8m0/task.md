## Context

OCP's MXFP4 format quantizes a contiguous **block of 32** values with a
single shared power-of-two scale (E8M0: an 8-bit unsigned exponent, no
mantissa) and represents every element inside the block in **E2M1** — one
sign bit, two exponent bits, one mantissa bit — an 8-level magnitude grid

$$
\mathcal{G} = \{0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6\}
$$

(16 signed codes total, since $\mathcal{G}$ is mirrored by the sign bit).
For a block $w = (w_1, \dots, w_{32})$, the shared exponent $e$ is the
smallest non-negative integer such that every scaled value fits inside
E2M1's representable range $[-6, 6]$:

$$
e = \max\!\left(0,\ \left\lceil \log_2\!\left(\frac{\max_i |w_i|}{6}\right) \right\rceil\right).
$$

Every element of the block is then divided by $2^{e}$ and **snapped to
the nearest signed E2M1 grid value**:

$$
\mathrm{code}_i = \operatorname{sign}(w_i) \cdot \operatorname*{arg\,min}_{g \in \mathcal{G}} \left|\, \frac{|w_i|}{2^{e}} - g \,\right| .
$$

Dequantization simply scales the codes back up:

$$
\hat w_i = \mathrm{code}_i \cdot 2^{e}.
$$

Every value in the block shares the same exponent $e$, so a single
outlier in a block forces every other value in that block onto a coarser
part of the E2M1 grid — the same "shared scale hurts quiet elements"
tradeoff that motivates smaller MX block sizes.

## Task

Implement `mxfp4_quant_dequant(weights)`:

```python
def mxfp4_quant_dequant(weights: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    ...
```

`weights` is a list of shape $(B, 32)$ — $B$ independent blocks of
32 values each. For every block:

1. Compute the shared exponent $e$ as defined above.
2. Snap every scaled element to the nearest signed E2M1 grid value to get
   that element's `code`.
3. Dequantize: `code * 2^e`.

Return a tuple `(codes, dequantized)`, both list of shape
$(B, 32)$:

- `codes`: the E2M1 grid value assigned to each element (one of the 16
  signed values in $\{0, \pm0.5, \pm1, \pm1.5, \pm2, \pm3, \pm4, \pm6\}$),
  **before** scaling back up.
- `dequantized`: `codes` scaled back up by $2^e$ (each block's own $e$).

## Example

```python

weights = [[0.0, 1.5, -2.3, 6.0] + [0.0] * 28]
codes, dequantized = mxfp4_quant_dequant(weights)

# max|w| = 6.0 -> e = ceil(log2(6/6)) = 0, so scale = 1 and codes are the
# elements snapped directly to the E2M1 grid:
# codes[0, :4] == [0.0, 1.5, -2.0, 6.0]
# dequantized[0, :4] == codes[0, :4]  (since 2^e == 1 here)
```

## What the gate checks

The gate loads a committed `weights.npy` fixture (50 blocks of 32
"weight"-like values, with block-to-block magnitude varying enough that
the shared exponent differs across blocks) and computes the reference
codes and dequantized values independently, exactly as specified above.
Your `codes` must match the oracle's **exactly**, element-for-element
(`exact_match == 1.0`) — codes live on a small set of exact powers-of-two
fractions, so there is no meaningful tolerance here: a different exponent
or a wrong nearest-grid-point choice produces a different code, not a
close one. Your `dequantized` values are additionally checked against the
oracle with `max_abs_err`, threshold $10^{-9}$ (this should be exact
floating-point arithmetic — `code * 2^e` — once the codes match).
