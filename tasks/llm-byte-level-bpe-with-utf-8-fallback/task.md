## Context

Byte-level BPE tokenizers operate on raw bytes instead of Unicode characters. A
text string is first encoded as UTF-8 bytes. This avoids requiring a separate
token for every possible character while still allowing arbitrary input.

Given a byte sequence $B = (b_1, b_2, \dots, b_n)$, byte-level BPE starts with
single-byte symbols and repeatedly merges adjacent symbol pairs according to a
ranked merge table. If a pair $(x, y)$ has the best available merge rank, it is
replaced by a new symbol until no ranked pairs remain.

A merge operation can be written as

$$
(x, y) \rightarrow xy .
$$

The final symbols are mapped to integer token ids through a vocabulary. Any
UTF-8 text can therefore be represented because the fallback alphabet contains
the original byte values.

## Task

Implement `byte_bpe_encode(text, vocab, merges)`.

Arguments:

- `text`: a Python string to tokenize.
- `vocab`: a dictionary mapping byte strings to integer token ids. It contains
  entries for individual bytes and merged byte sequences.
- `merges`: a dictionary mapping a pair of byte strings `(left, right)` to an
  integer priority. Lower priority values are applied first.

Return a list of integer token ids.

The algorithm must:

1. Encode `text` using UTF-8.
2. Start with one symbol per byte.
3. Repeatedly find the adjacent pair with the lowest merge priority among pairs
   present in `merges`.
4. Replace every occurrence of that selected adjacent pair with the concatenated
   symbol.
5. Convert the final byte symbols into ids using `vocab`.

## Example

```python
text = "hi🙂"

vocab = {
    b"h": 1,
    b"i": 2,
    b"\xf0\x9f\x99\x82": 20,
    b"hi": 30,
}

merges = {
    (b"h", b"i"): 0,
}

# byte_bpe_encode(text, vocab, merges)
# [30, 20]
```

## What the gate checks

The gate computes token ids with an independent byte-level BPE reference
implementation and compares the returned integer sequence using
`byte_exact_fraction`. The score must be exactly $1.0$, meaning every returned
token id must match the oracle output.

The tests include ASCII text, multibyte UTF-8 characters, emoji, and merge
chains. Returning Unicode-character tokens instead of UTF-8 byte fallback tokens
will fail.
