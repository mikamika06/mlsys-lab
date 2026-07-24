## Context

Paged attention systems store key-value cache tensors in fixed-size physical blocks instead of requiring one contiguous allocation per sequence.

A sequence has a logical block table that maps its logical blocks to physical blocks:

$$
L_i \rightarrow P_i
$$

where $L_i$ is the block position inside a sequence and $P_i$ is an allocated physical block id. When tokens are appended, new logical blocks are assigned physical blocks. When a sequence is freed, its physical blocks return to a free list and can be reused.

This task models a small part of a paged KV cache allocator. The allocator uses a block size of $16$ tokens and a LIFO free list. The most recently freed physical block is reused first.

## Task

Implement `PagedBlockAllocator` with these methods:

```python
class PagedBlockAllocator:
    def __init__(self, block_size=16):
        ...

    def append(self, seq_id, token_count):
        ...

    def free(self, seq_id):
        ...
```

The contract is:

- `append(seq_id, token_count)` appends tokens to a sequence. It must allocate enough new physical blocks so that the sequence has room for all tokens. Existing logical blocks are kept.
- The method returns a list of physical block ids allocated by this append call.
- `free(seq_id)` releases every physical block owned by the sequence and removes its logical-to-physical table entry.
- Physical blocks are assigned increasing ids when created.
- Reused blocks come from the free list in LIFO order.
- A physical block id can only belong to one sequence at a time.

The implementation should expose these attributes:

- `block_tables`: a dictionary mapping sequence ids to lists of physical block ids.
- `num_physical_blocks`: the total number of physical blocks ever created.

## Example

```python
alloc = PagedBlockAllocator()

alloc.append(1, 20)
# returns [0, 1]
# sequence 1 needs two blocks because 20 > 16

alloc.append(2, 16)
# returns [2]

alloc.free(1)

alloc.append(3, 10)
# returns [1]
# block 1 was freed last and is reused first
```

## What the gate checks

The gate replays several append/free traces against a reference allocator implementing the same rules. It checks that the returned allocation ids, final logical-to-physical tables, reuse order, and peak physical block count exactly match the reference behavior.
