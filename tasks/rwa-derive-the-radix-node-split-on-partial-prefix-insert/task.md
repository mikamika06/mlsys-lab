## Context

A radix tree stores a sequence of tokens by compressing chains of nodes into
single edges. When inserting a new sequence into an existing edge, the shared
prefix remains as the parent edge and the two different suffixes become child
edges.

For two edge sequences $a$ and $b$, the split position is the first index
where the tokens differ:

$$
k = \min \{ i \mid a_i \ne b_i \}.
$$

If all tokens in the shorter sequence match, the shorter sequence is a prefix of
the longer one and no partial-prefix split is needed.

For a partial-prefix insertion, one existing edge becomes a parent node and two
children are created: the old suffix and the inserted suffix. Therefore the
number of nodes increases by one compared with the original single edge.

## Task

Implement `derive_split(existing, incoming)`:

```python
def derive_split(existing: list[int], incoming: list[int]) -> tuple[int, int]:
    ...
```

The function receives the token sequence stored on an existing radix edge and a
new sequence being inserted.

Return a pair:

- `split_index`: the first index where both sequences contain different tokens.
  Return `-1` when one sequence is a prefix of the other and there is no partial
  split.
- `node_count`: the number of nodes after inserting the incoming sequence into a
  tree containing one existing edge node. A partial split creates one extra node,
  so return `2` for a partial split and `1` otherwise.

Use the prefix comparison algorithm directly. Do not sort the tokens or compare
sets because radix edges preserve token order.

## Example

```python
print(derive_split([1, 2, 3, 4], [1, 2, 9, 4]))
# (2, 2)

print(derive_split([1, 2, 3], [1, 2, 3, 4]))
# (-1, 1)
```

## What the gate checks

The gate builds several radix insertion cases and computes the expected result
with an independent prefix-walk oracle. The returned split position and node
count must exactly match the oracle output.
