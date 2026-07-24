## Context

Memory systems are affected by how programs traverse data. A cache stores data in
fixed-size lines, so nearby byte addresses can be reused efficiently when an
algorithm accesses data with good locality.

For a matrix with $n$ rows and $n$ columns stored in row-major order, the byte
address of an element is

$$
\mathrm{addr}(i,j) = \mathrm{base} + (i n + j)\cdot s,
$$

where $s$ is the element size in bytes.

A tiled traversal processes a matrix in smaller blocks. For a tile size $t$, the
access order visits blocks of rows and columns and then visits the elements
inside each block. This improves temporal and spatial locality because addresses
within a tile are reused before the cache must evict them.

The cache simulator models deterministic hardware behavior. A cache line holds
multiple nearby bytes, and the simulator tracks hits and misses based only on the
access address sequence.

## Task

Implement `blocked_access_trace(n, tile)`:

```python
def blocked_access_trace(n: int, tile: int) -> list[int]:
    ...
```

Return a list of byte addresses representing a traversal of an $n \times n$
matrix of 8-byte values. The traversal must cover every matrix element exactly
once.

The required traversal is a tiled row-major traversal:

1. Iterate tile rows from top to bottom.
2. Iterate tile columns from left to right.
3. Inside each tile, iterate rows first and columns second.
4. Convert each element coordinate $(i,j)$ into its byte address using
   $\mathrm{addr}(i,j) = (i n + j)\cdot 8$.

The final tiles at the edges may be smaller when $n$ is not divisible by `tile`.

## Example

```python
trace = blocked_access_trace(3, 2)

# The first tile contains:
# (0,0), (0,1), (1,0), (1,1)
# and addresses are:
# [0, 8, 24, 32, ...]
```

## What the gate checks

The gate runs the returned address trace through a deterministic cache simulator
with fixed parameters:

$$
\mathrm{line\_bytes}=64,\quad \mathrm{sets}=8,\quad \mathrm{ways}=2.
$$

The simulator computes the miss count from the trace. The returned trace must
match the reference tiled traversal exactly, so the cache behavior is also
deterministic.

A cache simulator is used instead of wall-clock timing so the result does not
depend on the machine running the task.
