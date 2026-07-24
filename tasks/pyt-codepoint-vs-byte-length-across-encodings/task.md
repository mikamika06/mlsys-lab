## Context

Python strings store Unicode text, while encodings convert that text into byte sequences. The number returned by `len(s)` is the number of Unicode code points in the string, not the number of bytes required by a particular encoding.

For a string $s$, its code point length is

$$L_{\mathrm{cp}}(s) = \mathrm{len}(s),$$

while an encoded byte length is

$$L_{\mathrm{bytes}}(s, e) = \mathrm{len}(s.\mathrm{encode}(e)).$$

These values can differ significantly. For example, ASCII characters take one byte in UTF-8, many BMP characters take multiple bytes, and astral Unicode characters require four bytes in UTF-8.

## Task

Implement `encoding_lengths(strings)`:

```python
def encoding_lengths(strings: list[str]) -> list[tuple[int, int, int]]:
    ...
```

For every string in `strings`, return a tuple containing:

1. The number of Unicode code points.
2. The number of UTF-8 encoded bytes.
3. The number of UTF-16 little-endian encoded bytes.

The output list must preserve input order. Use Python string operations and encoding APIs rather than manually counting Unicode ranges.

## Example

```python
values = ["abc", "é", "😀"]

print(encoding_lengths(values))
# [(3, 3, 6), (1, 2, 2), (1, 4, 4)]
```

## What the gate checks

The gate computes the expected values using Python's own Unicode implementation and encoding machinery:

$$
(\mathrm{len}(s), \mathrm{len}(s.\mathrm{encode}("utf-8")),
\mathrm{len}(s.\mathrm{encode}("utf-16-le")))
$$

for each test string.

The submitted function must return exactly the same list of tuples for ASCII text, BMP Unicode characters, and astral Unicode characters.
