## Context

Serving engines like vLLM cache the KV blocks of a prompt so that a second
request sharing the same prefix (a system prompt, a few-shot preamble, a
repeated document) can reuse the already-computed blocks instead of
recomputing them — a copy-on-write **prefix cache**. Tokens are grouped into
fixed-size blocks (`block_size` tokens each). A block is identified not just
by its own tokens, but by a **chained hash** over the whole prefix leading up
to it: block $i$'s identity depends on block $i-1$'s identity *and* block
$i$'s own tokens. This chaining is essential — hashing each block's tokens in
isolation (ignoring what came before) would incorrectly let two sequences
share a block whose tokens happen to match even though everything *before*
that block differs, corrupting the cache.

For a token sequence split into full blocks $B_0, B_1, \dots, B_{k-1}$ (any
trailing tokens that don't fill a full block are ignored — vLLM only caches
block-aligned prefixes), the chained hash is:
$$
h_0 = \mathrm{hash}\big((\,\mathrm{None},\, \mathrm{tuple}(B_0)\,)\big)
$$
$$
h_i = \mathrm{hash}\big((\,h_{i-1},\, \mathrm{tuple}(B_i)\,)\big) \quad \text{for } i \ge 1
$$
using Python's built-in `hash()` on the tuple. Two blocks (from the same or
different sequences, at the same or different block index) are the *same
physical block* iff their chained hash values are equal — which, by
construction, happens iff their full token history from position 0 is
identical up to and including that block.

Given a batch of sequences, the number of **physical blocks** actually
needed is the number of *distinct* hash values across all sequences (each
distinct hash is allocated once and shared by every sequence/position that
produces it). The **blocks saved** by sharing is the naive total (one block
per full block per sequence, no sharing) minus the number of physical
blocks.

## Task

Implement:

```python
def prefix_block_share(sequences: list[list[int]], block_size: int = 4) -> dict:
    ...
```

* `sequences` — a list of token-id sequences (each a `list[int]`).
* Returns a `dict` with:
  * `"block_hashes"` — a `list[list[int]]`, one list per input sequence,
    containing the chained hash (as defined above) of each **full** block in
    that sequence, in order. A sequence with `len(seq) // block_size == 0`
    full blocks contributes an empty list.
  * `"num_physical_blocks"` — `int`, the number of distinct hash values
    across all sequences' `block_hashes` lists combined.
  * `"blocks_saved"` — `int`, `(sum of len(block_hashes[s]) for all s) -
    num_physical_blocks`.

## Example

```python
seqs = [[1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 9, 9, 9, 9]]
out = prefix_block_share(seqs, block_size=4)
# out["block_hashes"][0][0] == out["block_hashes"][1][0]   -> True (same first block)
# out["block_hashes"][0][1] != out["block_hashes"][1][1]   -> True (diverge at block 1)
# out["num_physical_blocks"] == 3   (1 shared + 2 distinct second blocks)
# out["blocks_saved"] == 1          (4 naive blocks - 3 physical = 1 saved)
```

## What the gate checks

The grader builds a deterministic batch of sequences (fixed seed) with
overlapping prefixes of varying length (some sharing 2 full blocks, some
sharing none) plus a fully unrelated sequence, calls your function, and
compares — element for element — against an independent NumPy/Python oracle
implementing the exact chained-hash algorithm above: your `block_hashes`
lists, `num_physical_blocks`, and `blocks_saved` must all match exactly
(`exact_match` gate). A naive per-block hash that ignores the chain (hashes
each block's tokens alone) will falsely merge unrelated blocks that happen to
share tokens at the same position, or fail to merge blocks that really are
identical prefixes once a later divergence resets the chain — either way it
diverges from the oracle's counts.
