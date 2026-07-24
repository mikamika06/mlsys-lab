## Context

Memory is served in 64-byte **cache lines**. A row-major $n\times n$ `float64`
matrix stores element $(r,c)$ at linear index $r\cdot n + c$. Visiting indices
sequentially touches each line once (1 miss per 8 elements); jumping by a column
stride re-misses lines and is much slower.

## Task

Return `traverse(n)`: the order (a permutation of $0..n^2-1$) in which to visit
every element of a row-major matrix to **minimize cache misses**.

## Example

```python
traverse(2)  # -> [0, 1, 2, 3]  (row by row)
```

## What the gate checks

The cache simulator replays your access order and checks it covers every index
($\mathrm{covers\_all}=1$) with $\mathrm{misses}\le 512$ (only cache-friendly
orders pass).
