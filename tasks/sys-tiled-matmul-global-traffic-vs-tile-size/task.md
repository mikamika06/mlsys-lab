## Context

For a $n \times n \times n$ matmul $C = AB$ computed with square $T \times T$
tiles, one output tile is assigned per thread block. That block streams the
$K$ dimension in chunks of size $T$: at each step it stages one $T \times T$
chunk of $A$ and one $T \times T$ chunk of $B$ into fast (shared) memory,
reads each exactly once from global memory, and accumulates. There is no
reuse of a tile *across* output tiles (each block's fast-memory contents
disappear once it finishes), so the same region of $A$ gets re-fetched from
global memory once per output-tile row it participates in.

If $n$ is not a multiple of $T$, the grid still launches a whole number of
tiles per dimension, $\lceil n/T \rceil$ — the last tile is only partially
useful but the full $T\times T$ tile's worth of bytes is still moved. This
padding means **larger tiles are not always better**: a $T$ that leaves a
large remainder can move more total bytes than a smaller $T$ that divides
$n$ evenly.

## Task

Implement `tiled_matmul_traffic`:

```python
def tiled_matmul_traffic(n: int, elem_bytes: int, tile_sizes: list[int]) -> tuple[list[int], int]:
    ...
```

* `n` — matrix dimension ($A$, $B$, $C$ are all $n \times n$).
* `elem_bytes` — bytes per matrix element (e.g. `4` for `float32`).
* `tile_sizes` — a list of candidate square tile sizes $T$ to evaluate.

For each $T$, let $g = \lceil n / T \rceil$ (tiles per dimension). The
modeled global-memory traffic in bytes is:

$$
\text{traffic}(T) = \underbrace{2\, g^3\, T^2 \cdot \text{elem\_bytes}}_{A\text{ and }B\text{ tile reads, one per }(i,j,k)\text{ block}} \;+\; \underbrace{g^2\, T^2 \cdot \text{elem\_bytes}}_{C\text{ tile writes, one per }(i,j)\text{ block}}
$$

Return `(traffic, best_idx)`: `traffic` is a list of ints, one per entry of
`tile_sizes` (in the same order), and `best_idx` is the index into
`tile_sizes` of the $T$ minimizing `traffic(T)`.

## Example

```python
tiled_matmul_traffic(n=100, elem_bytes=4, tile_sizes=[8, 10, 12])
# T=8:  g=13 -> traffic = 1,168,128
# T=10: g=10 -> traffic =   840,000   <- smallest: 10 divides 100 exactly
# T=12: g=9  -> traffic =   886,464   <- WORSE than T=10 despite being bigger,
#                                         because ceil(100/12)=9 wastes a lot
#                                         of padding (9*12=108 vs n=100)
# returns ([1168128, 840000, 886464], 1)
```

## What the gate checks

A single gate, **modeled_mem_access**, requires that for every grading case
(several `(n, elem_bytes, tile_sizes)` triples, including ones where the
naive "bigger tile is always better" assumption picks the wrong answer):

* every entry of your returned `traffic` list exactly equals the model's
  value (integer equality — no tolerance), and
* your `best_idx` exactly equals the model's `argmin` over `tile_sizes`.

Any mismatch on any case fails the gate.
