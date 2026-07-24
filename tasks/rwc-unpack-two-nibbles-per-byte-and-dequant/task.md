## Context

Low-bit neural network storage formats often pack multiple values into one byte. In NF4 quantization, each stored value is a 4-bit code in the range $0$ to $15$. Two codes are stored in one `uint8` byte:

$$
\text{byte} = (\text{high\_code} \ll 4) \;|\; \text{low\_code}.
$$

The original floating-point value is reconstructed by looking up the NF4 level and multiplying by a block scale:

$$
w_i = \text{level}_{c_i} \cdot \text{absmax},
$$

where $c_i$ is the unpacked 4-bit code and `absmax` is the scale for the block.

The NF4 lookup table used by production implementations is a fixed set of 16 normalized levels:

$$
[-1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0,
0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7229, 1.0000].
$$

A packed byte array therefore represents twice as many weights as bytes.

## Task

Implement `unpack_nf4`:

```python
def unpack_nf4(packed: np.ndarray, absmax: float) -> tuple[np.ndarray, np.ndarray]:
    ...
```

The function receives a one-dimensional `uint8` NumPy array containing packed NF4 codes and a scalar block scale. It must return:

1. `codes`: a `uint8` array containing the unpacked codes in storage order. For each byte, the high nibble comes first and the low nibble comes second.
2. `weights`: a `float64` array containing the dequantized values using the NF4 lookup table and the supplied `absmax`.

The implementation should use NumPy operations for the unpacking and reconstruction.

## Example

```python
import numpy as np

packed = np.array([0x01, 0xFE], dtype=np.uint8)
codes, weights = unpack_nf4(packed, 2.0)

# codes:
# [0, 1, 15, 14]

# weights are:
# [
#   -2.0,
#   -1.3924,
#    2.0,
#    1.4458
# ]
```

## What the gate checks

The gate builds packed bytes and scales with a NumPy oracle. It checks that the returned unpacked codes exactly match the oracle output.

It also computes the maximum absolute error between the returned dequantized values and the oracle reconstruction:

$$
\max_i |\hat{w}_i - w_i|.
$$

The error must be at most $10^{-7}$, so both nibble order and NF4 level mapping must be correct.
