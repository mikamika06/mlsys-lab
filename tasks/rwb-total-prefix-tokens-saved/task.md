## Context

Prefix caching (SGLang's RadixAttention, vLLM's automatic prefix
caching) keeps every previously-prefilled request's tokens in a shared
tree, keyed by the token sequence itself. When a new request arrives, it
walks the tree from the root following its own tokens one at a time: as
long as the next token matches an existing edge, that token's KV cache
is already computed and can be **reused** ("saved"); the moment it
diverges (or runs off the end of the tree), every remaining token must
be freshly prefilled ("computed"). The new request's un-matched tail is
then grafted onto the tree so that future requests can reuse it too.

For a sequence $s = (s_1, \dots, s_L)$ inserted into a tree that already
contains some set of prior sequences, if the longest existing tree path
matching $s$'s own prefix has length $m \le L$, then

$$
\text{saved}(s) = m, \qquad \text{computed}(s) = L - m .
$$

## Task

Implement `prefix_tokens_saved`:

```python
def prefix_tokens_saved(seqs: list[list[int]]) -> tuple[int, int]:
    ...
```

`seqs` is a list of token-id sequences (each a `list[int]`), representing
requests arriving **in order**. Starting from an empty tree, insert them
one at a time:

1. Walk from the tree's root along `seq`, following existing edges for
   as long as they match `seq`'s own tokens exactly.
2. The number of tokens matched this way is `saved` for this sequence;
   the remaining tail is `computed`.
3. Graft that remaining tail onto the tree as new nodes/edges (so later
   sequences in the list can reuse it).

Return `(total_saved, total_computed)`, each summed over every sequence
in `seqs`.

## Example

```python
seqs = [
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],        # exact duplicate of the first
    [1, 2, 3, 4, 5, 6, 7],  # extends the first/second by 2 tokens
    [1, 2, 9, 9, 9],        # shares only the first 2 tokens
    [100, 200],             # shares nothing with the tree so far
]

prefix_tokens_saved(seqs)
# seq 1: tree is empty -> saved 0, computed 5
# seq 2: matches all 5   -> saved 5, computed 0
# seq 3: matches 5, then 2 new tokens -> saved 5, computed 2
# seq 4: matches [1, 2], then diverges -> saved 2, computed 3
# seq 5: matches nothing -> saved 0, computed 2
# totals: saved = 0+5+5+2+0 = 12, computed = 5+0+2+3+2 = 12
# -> (12, 12)
```

## What the gate checks

The gate, **exact_match**, runs your function against a reference
radix-tree simulation on a hand-built case (duplicate / extension /
partial-divergence / novel sequences), an empty-sequence edge case, and
several deterministically generated cases mixing shared "prompt prefix"
tokens with random suffixes. It compares the returned
`(total_saved, total_computed)` pair exactly — any off-by-one in where a
sequence diverges from the tree, or forgetting to graft the new tail
back on, fails that case.
