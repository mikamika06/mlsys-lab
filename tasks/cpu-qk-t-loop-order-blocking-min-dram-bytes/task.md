## Context

The attention mechanism's $QK^T$ contraction multiplies a query matrix
$Q \in \mathbb{R}^{S \times d}$ by the transpose of a key matrix
$K^T \in \mathbb{R}^{d \times S}$ to produce a score matrix
$S_{ij} = \sum_{k=0}^{d-1} Q_{ik} \cdot K_{jk}$.

Both $Q$ and $K$ are stored in **row-major** layout (each row is contiguous).
A naive $i$-$j$-$k$ loop reads $K$ with stride $d$ (column-major in $K^T$),
causing a cache miss on every access.

**Tiling** (blocking) reuses cache lines. By choosing a block size $B$ and
iterating in tiles of $B \times B$, we keep $Q[i, :]$ and $K[j, :]$ in cache
and dramatically reduce DRAM traffic.

$$\text{DRAM bytes} \approx \frac{S^2 \cdot d \cdot \text{element\_size}}{B}$$

## Task

Implement `qkt_access_order(S, d, B, elem_bytes) -> list[int]`, which returns
a flat list of **byte addresses** in the order they are accessed by a blocked
$QK^T$ computation:

- Outer loops: tile row $ii$ in steps of $B$, tile column $jj$ in steps of $B$
- Inner loops: row $i$ in $[ii, ii+B)$, then column $j$ in $[jj, jj+B)$, then
  $k$ in $[0, d)$
- For each $(i, j, k)$: emit address of $Q[i, k]$ then address of $K[j, k]$
- $Q$ row-major base = 0; $K$ row-major base = $S \times d \times \text{elem\_bytes}$

$$Q[i,k] \text{ addr} = (i \cdot d + k) \cdot \text{elem\_bytes}$$
$$K[j,k] \text{ addr} = S \cdot d \cdot \text{elem\_bytes} + (j \cdot d + k) \cdot \text{elem\_bytes}$$

## Example

```python
addrs = qkt_access_order(S=4, d=4, B=2, elem_bytes=4)
# Returns a list of byte addresses in tile-blocked QK^T access order
```

## What the gate checks

`check.py` simulates the returned address trace through a cache and checks that
`modeled_cache_misses` is at or below a budget — blocking must substantially
reduce misses compared to naive row-by-column access.
