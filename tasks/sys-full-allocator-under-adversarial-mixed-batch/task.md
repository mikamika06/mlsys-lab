## Context

Paged KV-cache allocators store transformer key/value state in fixed-size blocks rather than one contiguous buffer. A sequence is represented by logical token positions mapped onto physical blocks. When a request branches, multiple sequences can initially share the same physical blocks using reference counts. A write to a shared block requires copy-on-write allocation.

For a block with reference count $r$, releasing one owner changes the count by

$$
r \leftarrow r - 1 .
$$

A block can be evicted only when $r = 0$. The allocator must preserve the logical token values of every live sequence while reusing freed physical blocks.

This task models a simplified PagedAttention allocator. Blocks have a fixed capacity $B$ tokens. Each sequence has a logical token list and a block table mapping logical block indices to physical block ids.

## Task

Implement:

```python
def replay_kv_trace(trace, block_size, num_blocks):
    ...
```

The function receives a list of operations and returns the final allocator state.

Supported operations:

```python
("alloc", seq_id, tokens)
```

Create a new sequence with the given token list. Allocate enough blocks to store all tokens.

```python
("branch", new_seq_id, src_seq_id)
```

Create a new sequence by sharing all blocks from `src_seq_id`. Increase block reference counts.

```python
("append", seq_id, token)
```

Append one token to a sequence. If the target logical block is shared, perform copy-on-write before writing.

```python
("free", seq_id)
```

Delete a sequence and decrement all referenced block counts.

The returned value must be:

```python
(
    sorted_block_table,
    sorted_block_tokens,
    sorted_ref_counts,
)
```

where:

- `sorted_block_table` is a list of `(seq_id, [physical_block_ids])` pairs sorted by `seq_id`.
- `sorted_block_tokens` is a list of `(physical_block_id, tokens)` pairs sorted by physical block id. Only allocated blocks are included.
- `sorted_ref_counts` is a list of `(physical_block_id, ref_count)` pairs sorted by physical block id. Only allocated blocks are included.

Physical block ids must be reused by selecting the smallest available id first.

## Example

```python
trace = [
    ("alloc", 1, [10, 11, 12]),
    ("branch", 2, 1),
    ("append", 2, 13),
    ("free", 1),
]

result = replay_kv_trace(trace, 2, 4)
```

After the branch, both sequences share the first two-token block. The append to sequence `2` must copy that shared block before modifying it. The final result contains only sequence `2`, with its original tokens plus the appended token.

## What the gate checks

The gate contains adversarial mixed batches of allocation, branching, appending, and freeing. It builds an independent reference allocator from the same trace and compares the returned ownership tables, block contents, and reference counts with exact equality.

The reference implementation follows the allocator rules directly and does not use expected snapshots. A solution that skips reference counting, reuses non-free blocks, or mutates shared blocks will fail.
