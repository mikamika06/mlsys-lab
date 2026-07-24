## Context

Python strings store Unicode text, while encodings convert that text into byte
sequences. Different encodings can represent the same characters with different
bytes.

A valid round trip preserves the original string:

$$\mathrm{decode}(\mathrm{encode}(s, e), e) = s$$

where $s$ is a string and $e$ is an encoding.

The encoded bytes are also part of the behavior. For example, `utf-16` may add a
byte-order marker, while `utf-16-le` produces little-endian UTF-16 bytes without
that marker. A byte-exact implementation must select the requested encoding
precisely.

## Task

Implement `encode_decode_round_trip(strings)`.

The function receives a list of strings containing only characters supported by
all required encodings. Return a dictionary with exactly these keys:

- `"utf-8"`
- `"utf-16-le"`
- `"latin-1"`

Each value must be a list with one tuple per input string. Each tuple must contain:

```python
(encoded_bytes, decoded_string)
```

where `encoded_bytes` is produced using the exact encoding name and
`decoded_string` is produced by decoding those bytes with the same encoding.

Do not use `utf-16` in place of `utf-16-le`, because the byte order marker makes
the byte output different.

## Example

```python
result = encode_decode_round_trip(["Cafe", "café"])

result["utf-8"][1]
# (b'caf\xc3\xa9', 'café')

result["latin-1"][1]
# (b'caf\xe9', 'café')
```

## What the gate checks

The gate computes the expected bytes using Python's codec implementation as the
oracle. It compares all returned byte sequences with the oracle output using
`byte_exact_fraction`.

The required score is $1.0$, meaning every byte must match exactly.
