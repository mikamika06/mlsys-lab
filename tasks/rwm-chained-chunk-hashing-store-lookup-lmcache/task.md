## Context

Chunk-based cache systems can avoid recomputing work by identifying chunks that
already exist in a storage index. A position-aware rolling hash makes the chunk
identity depend on both the current content and all previous chunks.

For a sequence of chunks $c_1, c_2, \dots, c_n$, define the cumulative hash
sequence:

$$
h_i = H(h_{i-1} \Vert c_i),
$$

where $H$ is SHA-256, $\Vert$ is byte concatenation, and $h_0$ is the empty byte
string. The previous hash becomes part of the next hash input, so changing an
earlier chunk changes all later hash values.

A store maps previously computed hashes to the positions where those chunks were
seen:

$$
\text{store}: h_i \rightarrow \{p_1, p_2, \dots\}.
$$

A lookup returns the chunk indices and matching stored positions whose cumulative
hashes appear in the store.

## Task

Implement `lookup_reused_chunks(chunks, store)`:

```python
def lookup_reused_chunks(chunks: list[bytes], store: dict[bytes, list[int]]) -> list[tuple[int, int]]:
    ...
```

Compute the cumulative SHA-256 hash for each chunk in order. For every chunk
index $i$, if its cumulative hash is present in `store`, emit one tuple
`(i, position)` for every stored position associated with that hash.

Return the tuples sorted by chunk index and then by stored position. Do not
return duplicate tuples.

The initial previous hash is the empty byte string. Hash inputs must be the exact
byte concatenation of the previous digest and the current chunk.

## Example

```python
import hashlib

chunks = [b"a", b"b"]

h0 = b""
h1 = hashlib.sha256(h0 + b"a").digest()
h2 = hashlib.sha256(h1 + b"b").digest()

store = {
    h1: [10],
    h2: [20, 21],
}

result = lookup_reused_chunks(chunks, store)
# [(0, 10), (1, 20), (1, 21)]
```

## What the gate checks

The gate builds several stores from an oracle implementation of the chained
SHA-256 algorithm. The returned reuse index and position pairs must exactly
match the oracle output.

A solution that hashes each chunk independently, ignores previous chunk state, or
returns only the first matching position will fail because the cumulative hash
sequence and store matches will differ.
