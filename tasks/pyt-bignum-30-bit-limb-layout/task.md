## Context

CPython stores arbitrary precision integers as a sign and an array of base-$2^{30}$ digits on common 64-bit builds. Each digit is a limb containing a value in the range

$$
0 \le d_i < 2^{30}.
$$

For a non-negative integer $x$, the internal representation can be described as

$$
x = \sum_{i=0}^{k-1} d_i 2^{30i},
$$

where $d_0$ is the least significant limb.

A compact serialization of this representation can store each limb as a 32-bit little-endian unsigned integer. The unused top two bits of each stored word remain zero because each limb has only 30 meaningful bits.

Negative integers have the same limb magnitudes as their absolute value. The serialization format for this task is:

- one byte for the sign: `0` for non-negative values and `1` for negative values,
- four bytes containing the number of limbs as a little-endian unsigned integer,
- the limb values, each stored as a four-byte little-endian unsigned integer.

The zero value is represented with zero limbs.

## Task

Implement `pack_bignum(x)`:

```python
def pack_bignum(x: int) -> bytes:
    ...
```

Return the byte serialization of the CPython-style 30-bit digit layout described above. The function should work for positive, negative, and very large Python integers.

Do not convert the integer to a decimal string or use external serialization libraries. Extract the base-$2^{30}$ limbs directly with integer arithmetic.

## Example

```python
packed = pack_bignum(1 + (2 << 30))

# sign byte: 0
# limb count: 2
# limbs: [1, 2]
```

The resulting bytes are equivalent to:

```python
bytes([
    0,
    2, 0, 0, 0,
    1, 0, 0, 0,
    2, 0, 0, 0,
])
```

## What the gate checks

The gate builds reference bytes by reading CPython's active integer digit width from `sys.int_info` and reconstructing the base-$2^{30}$ limb representation algorithmically. Your output is compared byte-for-byte with this oracle using `byte_exact_fraction`.

The tested values include large positive and negative integers with limb boundaries and multiple carries. A result must achieve `byte_exact_fraction = 1.0`.
