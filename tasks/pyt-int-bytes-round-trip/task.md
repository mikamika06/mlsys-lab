## Context

Python integers have arbitrary precision, so converting an integer to raw bytes requires choosing a representation. A common representation stores a fixed number of bytes in little-endian order.

For an unsigned integer $x$, a width of $w$ bytes can represent values in the range

$$0 \le x < 2^{8w}.$$

The little-endian representation stores the least significant byte first. If the bytes are $b_0, b_1, \dots, b_{w-1}$, reconstruction follows

$$x = \sum_{i=0}^{w-1} b_i 2^{8i}.$$

Python exposes these operations through `int.to_bytes` and `int.from_bytes`.

## Task

Implement `int_bytes_round_trip(values, width)`.

The function receives a list of non-negative Python integers and an integer byte width. It must:

1. Serialize every integer using exactly `width` bytes.
2. Use little-endian byte order.
3. Return the list of serialized byte strings.
4. Decode each byte string back into integers internally to verify the round trip. The decoded values do not need to be returned.

The returned value must be a list of `bytes` objects. If an integer does not fit into the requested width, allow Python's normal `OverflowError` behavior.

## Example

```python
values = [1, 256, 65535]
out = int_bytes_round_trip(values, 2)

# out is:
# [b'\x01\x00', b'\x00\x01', b'\xff\xff']
```

## What the gate checks

The gate builds test cases and computes the reference serialization using Python's built-in integer byte conversion. The returned byte buffers are compared with the oracle output using `byte_exact_fraction`.

A score of $1.0$ means every serialized byte matches the reference exactly. Different byte order, variable-length encoding, or incorrect handling of fixed widths will fail.
