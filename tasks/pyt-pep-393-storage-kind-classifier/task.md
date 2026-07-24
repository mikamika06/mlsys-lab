## Context

CPython stores Unicode strings using the flexible representation from PEP 393. The internal storage uses the smallest character width that can represent every code point in the string.

For a string $s$, define the maximum code point:

$$m = \max_{c \in s} \operatorname{ord}(c)$$

The Unicode storage kind is determined by the width required for that maximum value:

- $1$ byte per character for Latin-1-compatible strings.
- $2$ bytes per character for strings requiring the Basic Multilingual Plane.
- $4$ bytes per character for strings requiring supplementary Unicode characters.

The storage width is an implementation detail of CPython and is not the same as the length of an encoded representation such as UTF-8.

## Task

Implement `classify_storage_kind(strings)`:

```python
def classify_storage_kind(strings):
    ...
```

The function receives a list of non-empty Python strings and returns a list of integers. Each integer must be the PEP 393 storage width used by the running CPython interpreter:

- `1` for one-byte storage.
- `2` for two-byte storage.
- `4` for four-byte storage.

The implementation should classify the internal representation, not the number of bytes produced by `encode()`.

## Example

```python
classify_storage_kind(["hello", "café", "Ā", "😀"])
# [1, 1, 2, 4]
```

## What the gate checks

The gate uses the running CPython interpreter as an oracle. It measures `sys.getsizeof` for the submitted strings and compares those measurements with calibration strings of the same length whose storage widths are known from CPython's own representation.

The returned vector must exactly match the oracle for strings containing ASCII, Latin-1, BMP, and supplementary Unicode characters.
