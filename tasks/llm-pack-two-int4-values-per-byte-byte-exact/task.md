## Context

Quantized neural network weights are often stored using fewer bits than standard integer
types. An int4 value uses $4$ bits, so it can represent an unsigned code in the range
$0$ to $15$. Two int4 codes can share one byte because

$$
4 + 4 = 8 .
$$

The byte is split into two nibbles. In this task, the packing convention is fixed:
the first int4 value is stored in the low nibble and the second int4 value is stored
in the high nibble. For two values $a$ and $b$, the packed byte is

$$
\mathrm{byte} = (a \mathbin{\&} 15) \;|\; ((b \mathbin{\&} 15) << 4).
$$

If the number of int4 values is odd, the final high nibble is filled with zero.

## Task

Implement `pack_int4(values)`:

```python
def pack_int4(values):
    ...
```

The input is a one-dimensional sequence of integer int4 codes. Every value satisfies
$0 \leq x \leq 15$. Return a NumPy array with dtype `uint8` containing the packed
bytes.

Values must be packed in order. For example, input values
`[a, b, c, d]` becomes bytes containing `(a, b)` and `(c, d)`.

Do not change the packing order.

## Example

```python
import numpy as np

values = np.array([1, 2, 15, 0], dtype=np.int64)
packed = pack_int4(values)

# packed contains:
# [33, 15]
```

The first byte is $1 + (2 << 4) = 33$ and the second byte is
$15 + (0 << 4) = 15$.

## What the gate checks

The gate builds a reference packed buffer using the specified nibble-packing algorithm
and compares the returned bytes against it. The `byte_exact_fraction` metric is the
fraction of identical bytes between the candidate output and the oracle output.

The score must satisfy

$$
\mathrm{byte\_exact\_fraction} = 1.0 .
$$

A numerically similar result is not sufficient because the byte representation itself
is the required output.
