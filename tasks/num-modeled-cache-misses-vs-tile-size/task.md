## Context

Matrix multiplication computes

$$
C_{ij} = \sum_{k=0}^{K-1} A_{ik}B_{kj}.
$$

Real GEMM implementations use blocking so that small matrix tiles stay in a fast
cache. A simple cache model can count memory traffic without depending on a
specific machine.

Assume a fully associative cache containing $C$ bytes with cache lines of $L$
bytes. Each matrix element is an 8-byte floating point value. A memory access
touches one cache line. On a miss, that line is loaded into the cache. If the
cache is full, the least recently used line is evicted.

For a square blocked multiplication with tile size $T$, the algorithm processes
tiles in the order:

$$
C_{ij} \mathrel{+}= A_{ik}B_{kj}
$$

where each of the three indices iterates over tile coordinates. Within a tile
multiplication, elements are accessed with the loop order:

```text
for i in tile rows:
    for k in tile depth:
        read A[i, k]
        for j in tile columns:
            read B[k, j]
            update C[i, j]
```

The modeled access count is the number of cache line loads caused by these
accesses. It is a deterministic approximation of DRAM line traffic.

## Task

Implement `modeled_access_count(n, cache_bytes, line_bytes, tile_sizes)`.

The function must return a dictionary mapping each integer tile size in
`tile_sizes` to the modeled number of cache line loads for multiplying two
$n \times n$ matrices using the blocked access pattern above.

The model assumptions are:

* matrices contain float64 values, so each element occupies 8 bytes;
* matrix `A` starts at byte offset $0$;
* matrix `B` starts immediately after `A`;
* matrix `C` starts immediately after `B`;
* matrices use row-major storage;
* cache replacement is least recently used;
* a cache line contains addresses from `line_start` through
  `line_start + line_bytes - 1$.

The function should return only integer counts.

## Example

```python
result = modeled_access_count(
    4,
    cache_bytes=128,
    line_bytes=64,
    tile_sizes=[2, 4],
)

# result has the form:
# {2: <line load count>, 4: <line load count>}
```

The exact values depend on the cache model parameters. The gate computes them
with the same deterministic model.

## What the gate checks

The gate runs several matrix sizes, cache configurations, and tile sizes. It
compares the returned dictionary against an internal reference implementation of
the cache simulator.

The metric `modeled_access_count` must be exactly $1.0$. Any incorrect cache
behavior, address calculation, blocking order, or replacement policy will fail.
