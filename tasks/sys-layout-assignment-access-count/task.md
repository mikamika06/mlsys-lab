## Context

Compilers and autotuners (XLA, TVM, Halide) choose a memory layout for
multi-dimensional tensors. The two canonical choices are **row-major** (C order)
and **column-major** (Fortran order). A good layout keeps the working set
contiguous in memory for the dominant access pattern, maximizing cache-line
utilization.

For an $m \times n$ matrix the linear address of element $(r, c)$ under each
layout is:

**Row-major:**
$$\mathrm{addr}(r, c) = r \cdot n + c$$

**Column-major:**
$$\mathrm{addr}(r, c) = c \cdot m + r$$

Given a sequence of $k$ accesses $\bigl((r_0,c_0),\,(r_1,c_1),\,\dots,\,(r_{k-1},c_{k-1})\bigr)$,
the **modeled sequential-access count** counts how many consecutive access pairs
touch *adjacent* memory locations:

$$S = \sum_{i=0}^{k-2} \mathbf{1}\!\Bigl[\bigl|\mathrm{addr}(r_{i+1},c_{i+1}) - \mathrm{addr}(r_i,c_i)\bigr| = 1\Bigr]$$

A higher $S$ means the layout keeps more accesses sequential — those accesses
hit the same or adjacent cache lines, so the memory subsystem can service them
with fewer DRAM rows opened. A layout with $S = k - 1$ is perfectly sequential;
$S = 0$ means no pair is adjacent.

## Task

Implement:

```python
def modeled_access_count(m: int, n: int, access_pattern: list[tuple[int, int]],
                         layout: str) -> int:
    """Return the modeled sequential-access count for the given layout.

    Args:
        m: number of rows in the matrix.
        n: number of columns in the matrix.
        access_pattern: ordered list of (row, col) index pairs.
        layout: "row" for row-major, "col" for column-major.

    Returns:
        Integer count of consecutive access pairs that are adjacent in memory.
    """
    ...
```

Use the address formulas above. Return `0` when the pattern has fewer than two
accesses. Do not assume the matrix is square.

## Example

```python
# 4×4 matrix, row-wise scan (left-to-right, top-to-bottom)
pattern = [(0,0),(0,1),(0,2),(0,3),
           (1,0),(1,1),(1,2),(1,3),
           (2,0),(2,1),(2,2),(2,3),
           (3,0),(3,1),(3,2),(3,3)]

modeled_access_count(4, 4, pattern, "row")  # → 15  (all adjacent)
modeled_access_count(4, 4, pattern, "col")  # → 0   (stride-4 jumps)
```

## What the gate checks

The gate runs ten test cases covering square and non-square matrices, row-wise
and column-wise scans, diagonal access, and short / single-element patterns. It
computes the reference count with the same address formulas (pure integer
arithmetic — no floating point, no external libraries) and checks **exact
equality** for every case. Averaging over all cases, the gate metric
`modeled_mem_access` must equal `1.0`.
