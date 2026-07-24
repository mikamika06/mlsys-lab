## Context

In many inference pipelines, weights and activations are quantized to low‑precision integer types to reduce memory footprint and accelerate computation. A common choice is 4‑bit unsigned integers (int4). Since a byte contains eight bits, we can store two int4 values in one byte by packing the high nibble and the low nibble.

Let $x_i \in \{0,\dots,15\}$ be an unsigned 4‑bit integer. The packed representation of two consecutive values $(x_{2k}, x_{2k+1})$ is

$$
p_k = (x_{2k} \ll 4) \;\;|\;\; x_{2k+1},
$$

where $\ll$ denotes a left shift and $|$ the bitwise OR. If the input length is odd, the final byte contains the last value in its high nibble and zeros in the low nibble.

Unpacking reverses this process:

$$
x_{2k} = (p_k \gg 4) \;\; &\; 0xF,\qquad
x_{2k+1} = p_k \;\; &\; 0xF,
$$

where $\gg$ is a right shift and $&$ the bitwise AND.

## Task

Implement two functions:

```python
def pack_int4(values: np.ndarray) -> np.ndarray:
    """Pack an array of unsigned 4‑bit integers into a uint8 byte array."""
```

and

```python
def unpack_int4(packed: np.ndarray, length: int) -> np.ndarray:
    """Unpack a packed uint8 byte array back to the original array of length `length`."""
```

Both functions must accept and return NumPy arrays with dtype `np.uint8`. The packing should handle arbitrary input lengths; if the number of values is odd, pad the final byte’s low nibble with zeros. Unpacking must recover exactly the first `length` 4‑bit integers.

## Example

```python
import numpy as np
from your_module import pack_int4, unpack_int4

values = np.array([3, 12, 7, 0, 15], dtype=np.uint8)
packed = pack_int4(values)
# packed == array([0x3C, 0x70, 0xF0], dtype=uint8)

unpacked = unpack_int4(packed, len(values))
print(unpacked.tolist())
# [3, 12, 7, 0, 15]
```

## What the gate checks

Two gates are applied:

1. **Byte‑exactness** – The packed byte array must be identical to a reference implementation computed by NumPy. This is measured with `byte_exact_fraction`, which must equal $1.0$.

2. **Round‑trip correctness** – Unpacking the packed bytes must recover exactly the original integer array. This is verified with an exact match metric (`exact_match`).

Both gates must pass for a solution to be accepted.
