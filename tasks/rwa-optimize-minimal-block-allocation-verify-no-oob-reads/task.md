## Context

Production inference systems often store large tensors or memory pages in fixed-size blocks. A batch of requested positions should allocate only the blocks that contain those positions. Allocating unnecessary blocks wastes memory, while incorrect block selection can cause invalid reads.

A block size $B$ maps an element position $x$ to a block id

$$
\mathrm{block}(x) = \left\lfloor \frac{x}{B} \right\rfloor .
$$

The valid offset inside that block is

$$
\mathrm{offset}(x) = x \bmod B .
$$

For a batch of positions $x_1, x_2, \dots, x_n$, the minimal allocation is the set of unique block ids containing all positions. A gather operation is safe only when every offset is in the range

$$
0 \leq \mathrm{offset}(x_i) < B .
$$

## Task

Implement `minimal_block_allocation(positions, block_size)`:

```python
def minimal_block_allocation(positions: list[int], block_size: int):
    ...
```

Return a tuple:

```python
(block_ids, gather_offsets)
```

where:

- `block_ids` is a list of allocated block ids in ascending order with no duplicates.
- `gather_offsets` is a list of offsets corresponding to each input position in the same order as `positions`.
- Each value in `gather_offsets` must be a valid slot inside its allocated block.
- The function must allocate the smallest possible block set that covers all requested positions.

Positions are non-negative integers and `block_size` is a positive integer.

## Example

```python
positions = [0, 3, 8, 9, 16]
block_size = 8

block_ids, gather_offsets = minimal_block_allocation(positions, block_size)

# block_ids == [0, 1, 2]
# gather_offsets == [0, 3, 0, 1, 0]
```

The position `9` belongs to block $1$ because

$$
\left\lfloor \frac{9}{8} \right\rfloor = 1
$$

and its in-block offset is $1$.

## What the gate checks

The gate computes the expected allocation using an independent oracle that derives the minimal block ids and offsets from the block mapping formula. It checks that the returned block set has the minimum number of blocks, that the block ids match the oracle ordering, and that every gather offset is inside an allocated block.

A solution that allocates all possible blocks, uses an incorrect block size, or creates out-of-range gather offsets will fail.
