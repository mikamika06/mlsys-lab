## Context

Python strings store Unicode text as a sequence of code points. UTF-8 is an encoding that
maps each code point to a sequence of one to four bytes.

For a code point $U$ in the ranges used by UTF-8, the byte layout is:

$$
\begin{aligned}
U &\leq 0x7F &&\rightarrow [U] \\
U &\leq 0x7FF &&\rightarrow [0xC0 \,|\, (U >> 6),\ 0x80 \,|\, (U \mathbin{\&} 0x3F)] \\
U &\leq 0xFFFF &&\rightarrow [0xE0 \,|\, (U >> 12),\ 0x80 \,|\, ((U >> 6) \mathbin{\&} 0x3F),\ 0x80 \,|\, (U \mathbin{\&} 0x3F)] \\
U &\leq 0x10FFFF &&\rightarrow [0xF0 \,|\, (U >> 18),\ 0x80 \,|\, ((U >> 12) \mathbin{\&} 0x3F),\ 0x80 \,|\, ((U >> 6) \mathbin{\&} 0x3F),\ 0x80 \,|\, (U \mathbin{\&} 0x3F)]
\end{aligned}
$$

The standard library already performs this conversion, but implementing the byte
construction manually is useful for understanding how Python text crosses the
boundary into byte storage.

## Task

Implement `utf8_encode(text)`:

```python
def utf8_encode(text: str) -> bytes:
    ...
```

The function must return the UTF-8 byte representation of `text` without calling
`str.encode`, `bytes.encode`, or similar built-in text encoders. Construct the result
manually from Unicode code points. The function must support ASCII characters,
multi-byte characters, and astral code points above $0xFFFF$.

## Example

```python
print(utf8_encode("A"))
# b"A"

print(utf8_encode("¢€😀"))
# b"\xc2\xa2\xe2\x82\xac\xf0\x9f\x98\x80"
```

## What the gate checks

The gate compares the produced bytes with CPython's UTF-8 encoding oracle on a
collection of strings containing ASCII, two-byte characters, three-byte characters,
and astral Unicode code points.

The `byte_exact_fraction` score must be $1.0$. The gate also rejects solutions that
use `str.encode` or equivalent encode method calls, because the task requires manual
byte construction.
