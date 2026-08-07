## Context

Ring all-reduce sums $N$ ranks' buffers using only nearest-neighbor
communication in a ring, at bandwidth cost independent of $N$. Each
rank's buffer of length $L$ is split into $N$ equal chunks of size
$L/N$. The algorithm runs in two phases, each exactly $N-1$
synchronized rounds:

**Reduce-scatter.** In round $s$ ($s = 0,\dots,N-2$), every rank $r$
receives one chunk from its left neighbor $r-1 \pmod N$ and **adds** it
into its own copy of that same chunk index. After all $N-1$ rounds,
rank $r$ ends up holding the fully-reduced (summed over all $N$ ranks)
version of exactly one chunk: the chunk at index $(r+1)\bmod N$.

**All-gather.** In round $s$ ($s=0,\dots,N-2$), every rank receives one
chunk from its left neighbor and **overwrites** (not adds) its own copy
of that chunk index with it, propagating each rank's completed chunk
around the ring. After $N-1$ rounds every rank holds every chunk fully
reduced, i.e. its whole buffer equals

$$
\text{result} = \sum_{i=1}^{N} \text{buffers}_i .
$$

## Task

Implement `ring_all_reduce(buffers)`:

```python
def ring_all_reduce(buffers: list[list[float]]) -> list[list[float]]:
    ...
```

- `buffers`: list of `N` 1-D arrays, all the same length `L`, with `L`
  divisible by `N` — rank `i`'s neighbors are `(i-1) % N` and
  `(i+1) % N`.

Simulate the two-phase reduce-scatter + all-gather ring algorithm
described above (`N-1` rounds each) and return a list of `N` arrays:
every rank's final buffer, each equal to the elementwise sum of all `N`
input buffers.

## Example

```python
buffers = [[1.0, 2.0, 3.0, 4.0],
           [10.0, 20.0, 30.0, 40.0],
           [100.0, 200.0, 300.0, 400.0],
           [1000.0, 2000.0, 3000.0, 4000.0]]
result = ring_all_reduce(buffers)
# every rank's buffer ends up [1111., 2222., 3333., 4444.]
```

## What the gate checks

The grader runs 8 randomly generated cases (`random`
seeded, `N` ranks between 2 and 5, chunk size between 1 and 3 per rank)
plus the fixed example above, and compares every returned rank's buffer
to the elementwise sum of all inputs computed directly with Python.
`max_abs_err <= 1e-6` across all ranks and all cases. A reduce-scatter
that overwrites instead of accumulates, or an all-gather that runs the
wrong number of rounds so some chunks never finish propagating, will
leave at least one rank's buffer only partially summed.
