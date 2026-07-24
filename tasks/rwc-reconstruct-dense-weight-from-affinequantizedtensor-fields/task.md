## Context

Quantized-tensor libraries (e.g. torchao's `AffineQuantizedTensor`) don't
store a dense float weight at all — they store a compact `int_data`
payload plus the affine parameters needed to reconstruct it on demand.
For 4-bit weights, `int_data` is typically **nibble-packed**: two 4-bit
codes share one `uint8` byte. Given only these low-level fields, you must
rebuild the dense weight exactly the way the library's `dequantize()`
does.

Byte $i$ of `int_data_packed` packs the codes for raveled (row-major)
positions $2i$ and $2i+1$:

$$
\text{code}(2i) = \text{byte}_i \mathbin{\&} \texttt{0x0F}, \qquad
\text{code}(2i{+}1) = (\text{byte}_i \gg 4) \mathbin{\&} \texttt{0x0F}.
$$

If the tensor has an odd number of elements, the final byte's high nibble
is unused padding and must not be read as a code.

Codes are grouped into consecutive runs of `group_size` raveled elements
(the last group may be shorter), each with its own **affine** dequant
parameters:

$$
\hat x_j = \bigl(\text{code}(j) - \text{zero\_point}[g(j)]\bigr) \cdot \text{scale}[g(j)],
\qquad g(j) = \left\lfloor \frac{j}{\text{group\_size}} \right\rfloor .
$$

## Task

Implement `reconstruct_dense_from_affine_quantized`:

```python
def reconstruct_dense_from_affine_quantized(int_data_packed, scale, zero_point, group_size, shape):
    ...
```

- `int_data_packed` — `uint8` array of length $\lceil n/2 \rceil$ where
  $n = \prod(\text{shape})$: nibble-packed 4-bit codes as described above.
- `scale`, `zero_point` — `float64` arrays of length
  $\lceil n / \text{group\_size} \rceil$: one affine pair per group, in
  group order.
- `group_size` — int.
- `shape` — the target dense shape.

Return a `float64` array of `shape`, dequantized element-by-element with
its group's `(scale, zero_point)`.

## Example

With `shape=(5,)` (odd $n=5$) and `group_size=2`: bytes 0 and 1 hold codes
for positions 0–3 (low/high nibbles each), and byte 2's low nibble holds
the code for position 4 — its high nibble is unused padding. Positions
0–1 use group 0's `(scale, zero_point)`, positions 2–3 use group 1's, and
position 4 (alone) uses group 2's.

## What the gate checks

The grader quantizes several real weight tensors with a genuine per-group
affine 4-bit scheme (including shapes with an odd element count, group
sizes that don't evenly divide the tensor, and an all-zero/constant
group), packs the codes into nibbles, and compares your reconstructed
dense array against the same unpack → dequant pipeline computed
independently:

$$
\max_i |\hat x_i - x_i| \le 10^{-8}.
$$

Swapping the low/high nibble order, misreading the padding nibble on an
odd-length tensor, using the wrong element's group index, or applying one
global `(scale, zero_point)` instead of per-group values will all produce
a large, easily detected deviation.
