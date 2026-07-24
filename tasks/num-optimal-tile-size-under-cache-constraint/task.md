## Context

Blocked matrix multiplication divides a GEMM operation into smaller tiles so that
working data can stay in a fast cache. For a square tile size $b$, a simplified
cache constraint is that three tiles must fit:

$$
3b^2s \le C,
$$

where $s$ is the number of bytes per matrix element and $C$ is the cache capacity
in bytes.

A simple traffic model counts the number of matrix element loads from slower
memory. For multiplying matrices with dimensions $m \times k$ and $k \times n$,
using square tiles of size $b$, the number of block rows and columns is

$$
r = \left\lceil \frac{m}{b} \right\rceil,\qquad
c = \left\lceil \frac{n}{b} \right\rceil,\qquad
t = \left\lceil \frac{k}{b} \right\rceil .
$$

The modeled DRAM traffic is the number of elements loaded for the left and right
tiles over all block multiplications:

$$
\mathrm{traffic}(b) =
rct \cdot b^2 + rct \cdot b^2 .
$$

Larger tiles usually reduce the number of block operations, but the cache
constraint limits the usable tile size.

## Task

Implement `optimal_tile_size(m, n, k, cache_bytes, element_bytes)`:

```python
def optimal_tile_size(m: int, n: int, k: int, cache_bytes: int, element_bytes: int) -> int:
    ...
```

Return the positive integer tile size $b$ that minimizes the modeled DRAM
traffic while satisfying

$$
3b^2 \cdot \text{element\_bytes} \le \text{cache\_bytes}.
$$

If multiple tile sizes have the same minimum traffic, return the largest tile
size.

Do not use external libraries.

## Example

```python
b = optimal_tile_size(1024, 1024, 1024, 196608, 8)
# 90
```

The exact result depends on the integer cache constraint and the traffic model.

## What the gate checks

The grader computes the optimal tile size independently by enumerating all valid
tile sizes using the same mathematical traffic model. It then compares the
modeled traffic of your returned tile size against the oracle optimum.

The reported metric is

$$
\frac{\mathrm{traffic}(\text{returned } b)}
{\mathrm{traffic}(\text{oracle } b)}
$$

and must satisfy $\le 1$. Invalid tile sizes fail the gate.
