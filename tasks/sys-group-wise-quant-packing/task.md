## Context

Inference quantization reduces model storage by replacing floating point values with
small integer codes and storing the information needed to reconstruct an
approximation. For a group of values $x_0, x_1, \dots, x_{g-1}$, symmetric
group-wise quantization chooses a scale

$$
s = \frac{\max_i |x_i|}{2^{b-1}-1},
$$

where $b$ is the number of bits per integer value. The quantized values are

$$
q_i = \operatorname{round}\left(\frac{x_i}{s}\right).
$$

This task uses $b=4$, so each $q_i$ is stored as an unsigned nibble after
offsetting by $8$. Two nibbles are packed into each byte.

The packed format stores metadata before the integer payload. For each group of
8 float32 values, store one little-endian float32 scale followed by 4 bytes
containing the eight packed nibbles. Groups are stored consecutively in input
order.

## Task

Implement `pack_groupwise_int4(x)`:

```python
def pack_groupwise_int4(x: np.ndarray) -> bytes:
    ...
```

The input is a one-dimensional NumPy array of float values. Its length is always
a multiple of 8. Convert it to float32 for quantization.

For each consecutive group of 8 values:

1. Compute the scale using the formula above with $b=4$. If the maximum absolute
   value is zero, use scale $1.0$.
2. Compute integer values with NumPy rounding and clip them to $[-8, 7]$.
3. Add $8$ to each integer to obtain nibbles in the range $[0, 15]$.
4. Pack nibble pairs as `(low_nibble) | (high_nibble << 4)`.
5. Append the scale as little-endian float32 bytes, then append the four packed
   bytes.

Return the complete byte blob.

## Example

```python
import numpy as np

x = np.array([0, 1, -1, 2, -2, 3, -3, 4], dtype=np.float32)
blob = pack_groupwise_int4(x)
```

The result begins with the four little-endian bytes of the computed scale,
followed by the four bytes containing the eight quantized nibbles.

## What the gate checks

The gate builds the reference blob using NumPy float32 operations and the
specified packing procedure. The returned bytes must have
$\mathrm{byte\_exact\_fraction}=1.0$ compared with the oracle output.

A layout mismatch, different rounding behavior, wrong scale dtype, or incorrect
nibble ordering will fail the gate.
