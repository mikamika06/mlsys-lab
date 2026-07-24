## Context

The C++ Standard Library `std::deque` is typically implemented as a
sequence of fixed-size memory blocks (a "map" of block pointers). This
lets it grow at both the front and the back without reallocating existing
elements.

A common implementation choice allocates blocks of a fixed byte size,
usually 512 bytes. The number of elements $N$ that fit into one block is
$\max(1, \lfloor 512 / \text{sizeof}(T) \rfloor)$.

Because elements can be pushed to the front, the first element (logical
index $0$) of the deque might not start at offset $0$ of its memory
block — instead it starts at some `first_offset`. To find the physical
location of the $i$-th element, map its logical index $i$ to a
`(block_index, block_offset)` pair based on this `first_offset` and the
block capacity $N$.

## Task

Implement, in `solve.cpp`,

```cpp
std::vector<std::pair<long, long>> deque_mapping(long elem_size, long first_offset,
                                                   const std::vector<long>& indices);
```

`elem_size` is `sizeof(T)` for the element type (already computed by the
real compiler — you do not need to derive it). You must:

1. Compute $N = \max(1, \lfloor 512 / \text{elem\_size} \rfloor)$
   (integer division).
2. For each index $i$ in `indices`, compute `absolute = first_offset + i`,
   then `block_index = absolute / N` and `block_offset = absolute % N`
   (both integer division/modulo).
3. Return the `(block_index, block_offset)` pairs, in the same order as
   `indices`.

## Example

For `elem_size = 8` (e.g. a `{char, int}` struct), $N = \max(1, 512 /
8) = 64$ elements per block. With `first_offset = 60`:

- index `0` → absolute `60` → `(0, 60)`
- index `3` → absolute `63` → `(0, 63)`
- index `4` → absolute `64` → wraps to `(1, 0)`

## What the gate checks

The fixed driver (`main.cpp`) runs five fixed cases, each with `elem_size`
taken from a real compiled struct's `sizeof` (so the ABI arithmetic is
never guessed, only the block/offset mapping is yours to get right), and
prints every resulting `(block, offset)` pair. The gate is an exact
string match (`exact_match == 1.0`) against the reference's printed
output: a wrong $N$, a swapped `div`/`mod`, or mishandling the
`first_offset` wrap-around all change at least one pair and fail the
gate.
