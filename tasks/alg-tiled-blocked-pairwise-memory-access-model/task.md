## Context

The pairwise Euclidean distance between two rows $a, b \in \mathbb{R}^d$ is
$$\lVert a - b\rVert^2 = \sum_{k=1}^{d}(a_k-b_k)^2.$$
A naïve implementation loops over all ordered pairs $(i,j)$ and recomputes the distance from scratch, causing many repeated loads of the same data.  
When the data is stored in contiguous memory, each cache line holds
$$L = \frac{\text{line\_size}}{8}$$
float64 elements.  A row therefore occupies $\lceil d/L\rceil$ cache lines.

A tiled (blocked) algorithm first loads a block of $B$ rows into the cache and then computes all pairwise distances between that block and every other block, reusing the loaded data as much as possible.  
The number of cache‑line loads for each strategy can be expressed exactly:

* **Naïve**: For every ordered pair $(i,j)$ we load both rows $A_i$ and $A_j$.
  $$\text{naive\_loads}=n^2 \cdot 2\,\bigl\lceil d/L\bigr\rceil.$$

* **Tiled**: Let $\text{size}_b$ be the number of rows in block $b$ (full blocks have size $B$, the last may be smaller).  
  For each pair of blocks $(b_i,b_j)$ we load all rows from both blocks once:
  $$\text{tiled\_loads}=\sum_{b_i}\sum_{b_j}
     2\,\bigl\lceil d/L\bigr\rceil \,(\text{size}_{b_i}+\text{size}_{b_j}).$$

These formulas are deterministic and independent of the underlying hardware.

## Task

Implement a function that returns the two cache‑line load counts described above.

```python
def pairwise_memory_access(n: int, d: int, B: int, line_size: int) -> tuple[int, int]:
    """
    Return (naive_loads, tiled_loads) for an n×d matrix of float64 values.
    """
```

The function must use only integer arithmetic and return the two counts as Python `int`s.

## Example

```python
>>> pairwise_memory_access(5, 3, 2, 64)
(50, 56)
```

Explanation:  
$L = 64/8 = 8$, so $\lceil d/L\rceil = \lceil 3/8\rceil = 1$.  
Naïve loads: $5^2 \cdot 2 \cdot 1 = 50$.  
Blocks: sizes are `[2, 2, 1]`.  
Tiled loads:
```
(0,0): 2*(2+2)=8
(0,1): 8
(0,2): 6
(1,0): 8
(1,1): 8
(1,2): 6
(2,0): 6
(2,1): 6
(2,2): 4
```
Sum = $56$.

## What the gate checks

The grader computes the reference counts using the formulas above and compares them exactly to your output.  Your solution must return a tuple of two integers that matches the reference for all test cases.  No other metrics are evaluated.
