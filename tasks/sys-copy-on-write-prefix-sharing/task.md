## Context

vLLM's PagedAttention stores each sequence's KV cache as a list of
fixed-size physical **blocks** (each holding `block_size` KV vectors),
addressed through a per-sequence **block table**. When several decoding
branches share the same prompt, they can literally share the *same*
physical blocks for that prompt instead of copying it $B$ times — as
long as none of them ever writes into a shared block. A **full** block
is immutable and is simply never written into again (the next token
always starts a fresh block). A **partial** last block, however, has
free room, so appending a token to it *would* be an in-place write. If
that block is still shared (its reference count $> 1$), an in-place
write would corrupt every other branch's view of it — so the writer
must first **copy-on-write (COW)**: allocate a new physical block,
copy the old block's contents into it, drop its own reference to the
old block, and write into the new one. Only once a writer holds the
*sole* remaining reference ($\text{refcount} = 1$) can it write in
place for free.

For a prompt of length $P$ and block size $S$, the prompt occupies
$\lfloor P/S \rfloor$ full blocks plus, if $P \bmod S \neq 0$, one
shared partial block holding the remaining $P \bmod S$ tokens.

## Task

Implement `cow_kv_branches(prompt_kv, branch_kvs, block_size)`:

```python
def cow_kv_branches(prompt_kv, branch_kvs, block_size):
    ...
```

- `prompt_kv`: `(P, D)` float array — KV vectors for the shared prompt.
- `branch_kvs`: list of `B` arrays; `branch_kvs[b]` has shape `(L_b, D)`
  — the KV vectors branch `b` appends after the shared prompt, in order.
- `block_size`: number of KV-vector slots per physical block.

All `B` branches start out referencing the *same* physical blocks that
hold `prompt_kv` (full blocks, plus the shared partial block if one
exists). Process branches in order `0..B-1`, and within a branch
process its tokens in order, applying the full/partial/COW rules from
the Context section above (a full block always gets a fresh block on
overflow; a partial block is written in place iff its current refcount
is `1`, otherwise it is copied first).

Return `(branch_sequences, total_blocks_allocated)`:

- `branch_sequences[b]` is the `(P + L_b, D)` reconstructed KV sequence
  for branch `b` — the shared prompt followed by its own appended
  tokens, in order, with the exact original values (no corruption from
  another branch's writes).
- `total_blocks_allocated` is the total count of *distinct* physical
  blocks ever allocated across the whole simulation: the initial
  prompt blocks, plus every COW copy, plus every fresh block.

## Example

```python
# P = 5, block_size = 4: one full block [0,1,2,3] + one shared partial
# block [4] (1/4 slots used, refcount = 2 since both branches share it).
prompt_kv = [[random.gauss(0, 1) for _ in range(3)] for _ in range(5)]
branch_kvs = [[[random.gauss(0, 1) for _ in range(3)] for _ in range(2)], [[random.gauss(0, 1) for _ in range(3)] for _ in range(2)]]
seqs, n_blocks = cow_kv_branches(prompt_kv, branch_kvs, block_size=4)

# branch 0 appends first: the partial block has refcount 2, so branch 0
# must COW it into a new block before appending its first token, then
# appends its second token into that same new block (now refcount 1, no
# further copy needed) -> +1 block.
# branch 1 appends next: the ORIGINAL partial block now has refcount 1
# (branch 0 released it), so branch 1 writes both of its tokens into it
# in place -> +0 blocks.
# total_blocks_allocated == 1 (full block) + 1 (original partial) + 1 (COW copy) == 3
assert n_blocks == 3
assert seqs[0].shape == (7, 3) and seqs[1].shape == (7, 3)
assert all(abs(a - b) < 1e-5 for row1, row2 in zip(seqs[0], [prompt_kv[i] for i in range(len(prompt_kv))] + [branch_kvs[0][i] for i in range(len(branch_kvs[0]))]) for a, b in zip(row1, row2))
assert all(abs(a - b) < 1e-5 for row1, row2 in zip(seqs[1], [prompt_kv[i] for i in range(len(prompt_kv))] + [branch_kvs[1][i] for i in range(len(branch_kvs[1]))]) for a, b in zip(row1, row2))
```

## What the gate checks

The grader runs the example above plus 8 deterministically generated
random cases (`random.Random` seeded; varying prompt length,
block size, branch count and per-branch append length, including
branches that append zero tokens) against an independent oracle that
simulates the same block manager and (a) reconstructs each branch's
sequence by plain concatenation of the true values, and (b) counts
blocks with its own COW simulation.

Two gates must both pass:

- `max_abs_err <= 1e-5` — every returned sequence must exactly
  reconstruct the branch's true KV values (no cross-branch corruption
  from a missed copy-on-write, and no stale/aliased data).
- `block_count_exact_match == 1.0` — `total_blocks_allocated` must
  match the oracle's count on every case. A solution that skips
  sharing entirely (e.g. always deep-copies the whole prompt per
  branch) or that never triggers COW (always writes in place) will
  reconstruct correct values but report the wrong block count, and
  will fail this gate.
