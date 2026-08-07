## Context

int4 group-affine quantization is the standard weight/KV-cache compressor
used by production inference stacks: split each row into contiguous
groups of `group_size` elements, and give each group its **own** 4-bit
(16-level) affine grid — a per-group scale and zero-point — instead of
one grid for the whole tensor. For a group of values $x$:

$$
\text{scale} = \frac{\max(x) - \min(x)}{15}, \qquad
\text{zero} = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{-\min(x)}{\text{scale}}\right),\, 0,\, 15\right)
$$

$$
\text{code}_i = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{x_i}{\text{scale}} + \text{zero}\right),\, 0,\, 15\right), \qquad
\hat{x}_i = (\text{code}_i - \text{zero}) \cdot \text{scale}
$$

(if $\max(x) = \min(x)$ the group is constant and $\hat{x} = x$ exactly —
there's nothing to lose). Smaller groups fit outliers more tightly but
mean more scale/zero pairs to store; this task is about getting the
per-group math itself exactly right, since even a single mis-grouped
element throws off every downstream computation that reads from the
quantized tensor — including attention, which quantized KV caches feed
into every decode step.

## Task

Implement `quantize_dequantize_int4_grouped`:

```python
def quantize_dequantize_int4_grouped(x: list[list[float]], group_size: int) -> list[list[float]]:
    ...
```

- `x`: a 2-D `float64` array of shape `(rows, cols)`.
- `group_size`: a positive `int` that evenly divides `cols`.

Split each row into `cols / group_size` contiguous groups along the
columns, quantize-then-dequantize each group independently with the
4-bit affine formulas above, and return the reconstructed array
$\hat{x}$, same shape as `x`.

## Example

```python

x = [[0.0, 5.0, 10.0, 15.0, 100.0, 101.0, 102.0, 103.0]]
quantize_dequantize_int4_grouped(x, group_size=4)
# group 1 = [0,5,10,15]: scale=1.0, exact grid points -> reconstructs exactly
# group 2 = [100,101,102,103]: scale=3/15=0.2, tight grid -> near-exact
# (a single group_size=8 call would instead spread one scale over the
#  huge 0..103 range, wasting most of the 16 levels on the gap between
#  the two clusters)
```

## What the gate checks

The grader builds a `(rows, cols)` tensor from a seeded Python generator
(Gaussian values with injected outliers, `cols` a multiple of 32, 64, and
128) and computes the reference dequantized tensor independently in
Python for each of `group_size in {32, 64, 128}` — same formulas, applied
per-row-per-group, never calling your function.

Two gates apply. `mse` is the worst-case mean squared error between your
returned tensor and the oracle's, across all three group sizes (must be
`<= 1e-8` — this is a near bit-exact check on the per-group scale/zero
math and group boundaries, not a compression-quality bound). `attn_max_abs_err`
plugs your quantizer into a small **end-to-end** sanity check: it
quantizes a separate seeded `K` and `V` (`group_size=32`), runs standard
scaled dot-product attention with a fixed `Q` against both your quantized
`K`/`V` and the oracle's, and requires the max elementwise output
difference to stay `<= 0.3` — catching a quantizer whose grouping is so
wrong it makes the attention output unusable, even in the unlikely case
its raw MSE looked plausible.
