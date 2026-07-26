## Context

Byte‑Pair Encoding (BPE) is a simple compression technique that repeatedly merges the most frequent adjacent pair of symbols in a sequence.  
In this task we use a *very* small variant: we treat each character as its UTF‑8 byte value and merge consecutive identical bytes into a single token consisting of the byte value and a repetition count.

Formally, for a string $s$ let $\mathbf{b} = (b_1,\dots,b_n)$ be the sequence of its UTF‑8 byte values.  
The encoding produces a list of pairs
$$
\bigl((v_k,c_k)\bigr)_{k=1}^{m},
$$
where each $v_k \in \{0,\dots,255\}$ is a byte value and $c_k$ is the number of consecutive occurrences of that byte in $\mathbf{b}$.  
The decoding simply expands each pair back into $c_k$ copies of $v_k$.

This representation preserves the original string exactly while grouping runs of identical bytes together. It is a toy example of BPE that is easy to reason about and implement.

## Task

Implement two functions:

```python
def bpe_encode(text: str) -> list[tuple[int, int]]:
    """Encode *text* into a list of (byte_value, count) pairs."""
```

```python
def bpe_decode(tokens: list[tuple[int, int]]) -> str:
    """Decode the token list produced by `bpe_encode` back to the original string."""
```

Both functions must work for any Unicode string.  The encoded tokens should be a Python list of tuples; each tuple contains two integers: the byte value (0‑255) and the run length (≥ 1).  
The decoder must return exactly the input string.

## Example

```python
>>> s = "aaabbbcc"
>>> t = bpe_encode(s)
>>> t
[(97, 3), (98, 3), (99, 2)]
>>> bpe_decode(t)
'aaabbbcc'
```

For a string containing non‑ASCII characters the UTF‑8 bytes are used:

```python
>>> s = "😀😃"
>>> t = bpe_encode(s)
>>> t
[(240, 1), (159, 1), (152, 1), (128, 1), (240, 1), (159, 1), (152, 1), (131, 1)]
>>> bpe_decode(t) == s
True
```

## What the gate checks

Two gates. The first ensures that decoding the encoded tokens reproduces the original string byte‑exactly; this is measured by `arena.scorers.byte_exact_fraction` and must equal `1.0`.  
The second verifies that the token list produced by `bpe_encode` matches a reference implementation exactly, element‑wise; this uses a custom scorer that compares lists of tuples and also requires a fraction of `1.0`.

Both gates together guarantee that the functions are not only a round‑trip but also produce the correct token representation.
