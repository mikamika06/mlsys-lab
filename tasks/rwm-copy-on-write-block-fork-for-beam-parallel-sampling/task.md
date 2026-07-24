## Context

Beam search and parallel sequence generation often fork a sequence into multiple
candidates. Copying every token block during a fork wastes memory, so production
systems commonly use copy-on-write (COW) storage.

A sequence is stored as references to fixed-size blocks. A fork creates a new
sequence that points to the same full blocks and increments each block's reference
count. When a sequence later writes to a block with multiple owners, that block
must be copied before modification. This preserves isolation while avoiding
unnecessary copies.

If a block has reference count $r$, a write is allowed in place only when

$$r = 1.$$

Otherwise, the operation creates a new block containing the same values, changes
the sequence to reference the new block, and updates reference counts.

## Task

Implement `simulate_cow_blocks(block_size, initial, ops)`.

The arguments are:

- `block_size`: positive integer block capacity.
- `initial`: list of integer tokens for the first sequence.
- `ops`: list of operations. Each operation is a tuple:
  - `("fork", sequence_id)` creates a new sequence forked from the given sequence.
    New sequence ids are assigned in increasing order starting at $1$.
  - `("append", sequence_id, token)` appends one token to a sequence.

The initial sequence has id $0$.

Return a dictionary with exactly these keys:

- `"blocks"`: a list of physical blocks, where each block is a list of tokens.
- `"refs"`: the reference count for every physical block.
- `"seqs"`: a list where each entry contains the block ids used by that sequence.

The implementation must model block sharing and copy-on-write behavior. The returned
block ids must match the order in which physical blocks are allocated.

## Example

```python
result = simulate_cow_blocks(
    4,
    [1, 2, 3, 4],
    [
        ("fork", 0),
        ("append", 1, 5),
        ("append", 0, 6),
    ],
)

# result describes:
# sequence 0 using a copied block after appending,
# sequence 1 keeping the original shared block.
```

## What the gate checks

The gate replays fork and append traces against an independent reference
implementation of the same COW block model. It checks that the returned
physical blocks, reference counts, and sequence-to-block mappings exactly match
the oracle.

A solution that always copies on fork, mutates shared blocks directly, or ignores
reference counts will fail because the physical allocation history differs.
