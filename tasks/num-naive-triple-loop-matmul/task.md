## Context

Matrix multiplication of two real matrices $A \in \mathbb{R}^{m\times k}$ and $B \in \mathbb{R}^{k\times n}$ is defined by
$$C_{ij} = \sum_{p=1}^{k} A_{ip}\, B_{pj},\qquad i=1,\dots,m,\; j=1,\dots,n.$$

The naive algorithm evaluates each entry of $C$ independently using a triple loop over $i$, $j$, and $p$. This is the most straightforward implementation but has cubic time complexity.

## Task

Implement the function `naive_matmul(A, B)` that takes two 2‑D lists `A` and `B`, verifies that their inner dimensions agree, and returns a new list of shape `(m,n)` containing the matrix product. The result must consist of floats. Do **not** use the `@` operator; instead write an explicit triple loop.

```python
def naive_matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    ...
```

## Example

```python
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = naive_matmul(A, B)
# C should be:
# array([[19., 22.],
#        [43., 50.]])
```

## What the gate checks

The grader computes a Python reference product `[[sum(a * b for a, b in zip(r, c)) for c in zip(*B)] for r in A]` and compares it to your output using the scorer
`max_abs_err`. The absolute error must satisfy

$$\max_{i,j} |C_{ij}^{\text{your}} - C_{ij}^{\text{ref}}| \le 10^{-6}.$$

Any deviation larger than this threshold causes the gate to fail. The implementation is also expected to use explicit Python loops; using vectorized Python operations or BLAS calls will still produce a correct result but is discouraged for this exercise.
