## Context

A cache or reuse system often identifies repeated work by hashing input segments. A common mistake is to hash every segment independently:

$$h_i = H(x_i, x_{i+1}, \dots, x_{i+k-1}).$$

This ignores where the segment appears in the larger sequence. If the same local chunk occurs in a different prefix context, an independent chunk hash can report a false reuse opportunity.

A position-aware chained hash includes the previous state when computing each next state:

$$s_{i+1} = (B \cdot s_i + x_i) \bmod M.$$

A chunk starting at position $i$ is identified using both its previous chain state and its contents:

$$K_i = (s_i, x_i, x_{i+1}, \dots, x_{i+k-1}).$$

Only chunks with the same key represent safe reuse candidates.

## Task

The provided implementation of `find_reusable_chunks` is incorrect. It hashes each chunk in isolation, which can mark the same chunk at another location as reusable even when its prefix context is different.

Fix `find_reusable_chunks(trace, chunk_size)` so it returns the starting indices of chunks that can safely reuse a previously seen computation.

The function signature is:

```python
def find_reusable_chunks(trace: list[int], chunk_size: int) -> list[int]:
    ...
```

Return a list of zero-based chunk start indices in increasing order. A chunk is reusable only if an identical chunk with the same prefix chain state appeared earlier in the trace.

## Example

```python
trace = [4, 9, 2, 7, 9, 2, 7]
find_reusable_chunks(trace, 3)
# []
```

The chunk `[9, 2, 7]` appears twice, but the second occurrence has a different prefix state, so it is not a safe reuse.

For a trace where a repeated chunk occurs after an identical prefix state, the later index is returned.

## What the gate checks

The gate builds traces with repeated local chunks and computes the expected result using the chained prefix-hash algorithm described above. The candidate implementation must match the oracle output exactly.

The test rejects independent chunk hashing because it produces false reuse hits on chunks that only match locally.
