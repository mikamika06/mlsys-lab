## Context

Quantized models often store small integer codes by packing several values into
machine words. For 4-bit quantization, each code $c_i$ satisfies

$$0 \le c_i < 2^4 = 16.$$

The MLX storage layout used in this task packs blocks of $64$ codes into $8$
32-bit unsigned words. Each word stores $8$ consecutive nibbles. For a block of
codes $c_0, \dots, c_{63}$, word $w_j$ contains

$$
w_j = \sum_{k=0}^{7} c_{8j+k} 2^{4k}.
$$

The lowest four bits of each word contain the first code in that group, and the
highest four bits contain the eighth code. This layout allows exact
reconstruction because each nibble is stored independently.

## Task

Implement these functions:

```python
def pack_4bit_codes(codes: np.ndarray) -> np.ndarray:
    ...

def unpack_4bit_codes(words: np.ndarray) -> np.ndarray:
    ...
```

`pack_4bit_codes` receives a one-dimensional NumPy array of unsigned integer
codes. The length is always a multiple of $64$, and every value is in the range
$[0, 15]$. It must return a `np.uint32` array containing the packed MLX words.

`unpack_4bit_codes` receives the packed `np.uint32` words and must return the
original codes as a one-dimensional NumPy array with dtype `np.uint8`.

The implementation must preserve values exactly. Use bit operations rather than
converting through strings or text representations.

## Example

```python
import numpy as np

codes = np.array([0, 1, 2, 3, 4, 5, 6, 7] + [0] * 56, dtype=np.uint8)

words = pack_4bit_codes(codes)

# words[0] stores the first eight nibbles:
# 0x76543210

restored = unpack_4bit_codes(words)

# restored is identical to codes
```

## What the gate checks

The gate builds a NumPy oracle implementation of the MLX packing layout. It
compares the returned packed `uint32` words against the oracle bit-for-bit and
then verifies that unpacking the words reconstructs the original codes.

The `exact_match` score must equal $1.0$. Any different nibble order, incorrect
dtype, or lossy unpacking fails the gate.
