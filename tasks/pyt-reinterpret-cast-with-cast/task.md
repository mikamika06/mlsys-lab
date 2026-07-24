## Context

Python's buffer protocol allows objects to expose raw memory without copying.
A `memoryview` can reinterpret the same storage using `.cast()`.

A cast changes the interpretation of the bytes rather than converting values.
For example, a four-byte sequence can represent either a 32-bit integer or a
32-bit floating point number depending on the active view:

$$
\text{same bytes} \rightarrow \text{integer view} \rightarrow \text{float view}
$$

The underlying byte representation should remain unchanged during a
reinterpretation.

## Task

Implement `reinterpret_roundtrip(data)`:

```python
def reinterpret_roundtrip(data: bytes) -> bytes:
    ...
```

The input is a bytes object with a length divisible by $4$. Reinterpret the
buffer as a view of C integers (`'i'`), then reinterpret the same storage as a
view of C floats (`'f'`), and return the bytes from the final view.

Use `memoryview.cast()` for the reinterpretation. Do not numerically convert the
values and do not modify the original buffer.

## Example

```python
raw = bytes([0, 0, 128, 63, 0, 0, 0, 64])

out = reinterpret_roundtrip(raw)

# out == raw
```

The bytes above encode two floating point values when interpreted as `'f'`, but
the required output is the original byte sequence.

## What the gate checks

The gate builds a reference result using Python's buffer protocol operations and
compares the candidate bytes with that oracle.

The score is `byte_exact_fraction`, defined as the fraction of bytes that match.
A passing implementation must achieve $1.0$, meaning that every output byte
matches the oracle.
