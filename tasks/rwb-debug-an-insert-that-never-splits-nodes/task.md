## Context

A radix tree stores a set of token sequences by merging common prefixes. Unlike a
plain trie, a radix tree stores runs of tokens on edges. When inserting a sequence,
a divergence inside an existing edge must split that edge so the shared prefix is
kept.

For a set of sequences $S$, define the uncompressed token storage as

$$
C_{\mathrm{plain}} = \sum_{s \in S} |s|.
$$

The compressed radix tree stores only the tokens appearing on its edges. If the
total number of stored edge tokens is $C_{\mathrm{radix}}$, the saved amount is

$$
\mathrm{saved} = C_{\mathrm{plain}} - C_{\mathrm{radix}}.
$$

A buggy insert implementation can match only complete existing edges. When a new
sequence diverges in the middle of an edge, it incorrectly adds another branch
instead of splitting the edge, causing less prefix reuse.

## Task

Debug `total_saved_tokens(seqs)` so that it returns the number of tokens saved by
a correctly implemented radix tree.

The input `seqs` is a list of token sequences. Each sequence is a list of integers.
The function should insert every sequence into a radix tree, split existing edges
when necessary, and return the saved token count as an integer.

The provided implementation contains an insertion bug. Fix the implementation
without changing the function signature.

```python
def total_saved_tokens(seqs):
    ...
```

## Example

```python
seqs = [
    [1, 2, 3, 4],
    [1, 2, 5],
]

# Plain storage: 7 tokens.
# A radix tree stores edges [1,2], [3,4], and [5]:
# compressed storage: 5 tokens.
# saved tokens: 2.
```

## What the gate checks

The gate builds the expected result by running an independent radix tree oracle
that performs real edge splitting. The submitted function is run on several
sequence collections, and its returned `saved` value must exactly match the
oracle result.

The gate metric is `exact_match`. A value of $1.0$ means every test collection
returned the same saved-token count as the reference radix tree.
