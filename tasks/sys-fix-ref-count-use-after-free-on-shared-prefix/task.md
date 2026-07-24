## Context

PagedAttention-style inference systems store key/value tokens in fixed-size blocks. Multiple sequences may share a prefix of tokens, so physical blocks need ownership tracking.

A block with reference count $r$ is still reachable by $r$ logical owners. Freeing a block is only valid when

$$r = 0.$$

If a sequence releases a shared prefix block while another sequence still points to it, a use-after-free occurs: later reads observe missing or overwritten KV entries.

This task models a simplified paged KV cache. Each block stores a list of token ids. The allocator must update reference counts when sequences fork, append tokens, and delete sequences.

## Task

Implement `replay_kv_trace(ops, block_size)`.

The input `ops` is a list of operations. Each operation is a tuple:

- `("create", sequence_id, tokens)` creates a sequence with an initial token list.
- `("fork", new_sequence_id, parent_sequence_id)` creates a new sequence sharing all current blocks with the parent.
- `("append", sequence_id, token)` appends one token to a sequence.
- `("delete", sequence_id)` removes a sequence and releases its blocks.

The function must return a dictionary mapping every surviving sequence id to its final logical token list.

Blocks are internal implementation details. The implementation must correctly handle shared prefixes by maintaining ownership. When a shared block is modified, use copy-on-write behavior so other sequences keep their original tokens.

## Example

```python
ops = [
    ("create", "a", [1, 2, 3, 4]),
    ("fork", "b", "a"),
    ("append", "a", 5),
    ("delete", "a"),
]

result = replay_kv_trace(ops, 4)
# {"b": [1, 2, 3, 4]}
```

The append to sequence `"a"` must not corrupt `"b"` because the shared block is copied before modification.

## What the gate checks

The gate replays several traces through a reference allocator that models block ownership and copy-on-write behavior. The returned logical KV contents must exactly match the oracle result.

A solution that frees a block while another sequence still references it will produce incorrect logical tokens and fail the exact match gate.
