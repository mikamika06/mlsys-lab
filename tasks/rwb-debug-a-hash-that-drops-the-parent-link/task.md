## Context

In streaming data processing, a common task is to detect duplicate or repeated blocks of tokens (e.g., code snippets, byte sequences, or text n-grams) within a sliding window. A naive approach hashes the block content alone. But two occurrences of the same block content at different positions in the stream are **not** necessarily duplicates for the purpose of a "prefix-aware" cache — the preceding context matters. A *parent-linked* hash (often called a *position-dependent* or *prefix-aware* hash) mixes the identity of the immediately preceding block into the hash of the current block. This makes the hash of a block depend not only on its own tokens but on where in the stream it appears.

Formally, let the stream be a sequence of token blocks $B_1, B_2, \dots, B_n$, where each block $B_i$ is a tuple of integer token IDs. A *parent-linked salted hash* is computed as:

$$
h_i = H( \; B_i, \; \text{salt}, \; h_{i-1} \; )
$$

where $H$ is a mixing function (e.g., a cyclic-redundancy-like polynomial or a simple Python hash of the tuple `(B_i, salt, h_{i-1})`), and $h_0 = \text{salt}$ (or some initial seed). Without the parent link $h_{i-1}$, two identical blocks $B_i = B_j$ produce the same hash regardless of their position, which leads to false hits when looking up repeated content under different prefixes.

## Task

You are given a **buggy** implementation of a `block_salted_hash` function. The function is supposed to produce a prefix-aware hash by chaining the previous block's hash into the current block's hash. However, the buggy version **drops the parent link** — it hashes only the block tokens and the salt.

The correct (oracle) algorithm computes:

```text
hash = salt
for each block in the stream (in order):
    hash = hash_of( (block_tokens, salt, hash) )
```

Implement the correct algorithm in `solution_ref.py`. The learner is given the buggy version in `starter.py` and must **fix** it so that `block_salted_hash` returns the correct mapping from each block's position to its prefix-aware hash.

The function signature:

```python
def block_salted_hash(stream: list[tuple[int, ...]], salt: int) -> list[int]:
    ...
```

`stream` is a list of blocks, each block is a tuple of integer token IDs. `salt` is an integer seed. The function returns a list of integer hashes, one per block, in order.

## Example

```python
stream = [
    (1, 2),     # block 0
    (3, 4),     # block 1
    (1, 2),     # block 2 — same tokens as block 0
]
salt = 42

# Correct output (prefix-aware):
# The hash of block 2 differs from block 0 because block 1's hash was chained in.
result = block_salted_hash(stream, salt)
print(result)  
# e.g., [436905508, 1489342547, -1934286420]  (values depend on Python's hash)
# Key point: result[0] != result[2]
```

The buggy version would produce `result[0] == result[2]` because it ignores the parent link.

## What the gate checks

Exact match: the entire list of hashes returned by the learner's function must exactly equal (element-wise) the list produced by the correct oracle (which chains the parent link). The grader re-runs the learner's `block_salted_hash` on a fixed set of test streams (including streams with identical blocks under different prefixes) and compares the output to the oracle's output. The gate `exact_match` must be 1.0.
