## Context

Wanda pruning uses an importance score that combines the magnitude of a weight with an input activation statistic. For a weight matrix $W \in \mathbb{R}^{m \times n}$ and input calibration matrix $X \in \mathbb{R}^{k \times n}$, define the per-input Wanda score matrix

$$
S_{ij} = |W_{ij}| \cdot \sqrt{\frac{1}{k}\sum_{t=1}^{k} X_{tj}^{2}} .
$$

The $n$ input columns are divided into consecutive groups of four. A 2:4 structured sparsity pattern keeps exactly two entries in every group of four inputs for each output row. This task uses a variant where the two largest Wanda scores are kept independently inside every group.

The resulting binary mask $M$ has the same shape as $W$:

$$
M_{ij} =
\begin{cases}
1 & \text{if weight } W_{ij} \text{ is selected}\\
0 & \text{otherwise.}
\end{cases}
$$

For each row and each group of four columns, exactly two positions with the largest values of $S$ must have mask value $1$.

## Task

Implement `wanda_2_4_mask(W, X)`:

```python
def wanda_2_4_mask(W, X):
    ...
```

The function receives:

- `W`: a list of lists of floats with shape $(m, n)$.
- `X`: a list of lists of floats with shape $(k, n)$.
- $n$ is divisible by $4$.

Return an integer list of shape $(m, n)$ containing only zeros and ones.

Compute the Wanda scores, then for every output row and every consecutive block of four input columns keep the two highest-scoring entries. If scores are tied, the lower column index inside the group must be preferred.

## Example

```python

W = [[1.0, -4.0, 2.0, 3.0]]
X = [
    [1.0, 2.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
]

M = wanda_2_4_mask(W, X)
# array([[0, 1, 1, 0]])
```

The scores are proportional to $[1, 4, 2, 3]$, so the two largest values in the group are the second and third entries.

## What the gate checks

The gate computes a Python oracle from the Wanda score definition and compares the returned mask exactly. It also verifies the structural constraint that every consecutive group of four inputs in every output row contains exactly two surviving entries.

A solution passes only if it produces the same 2:4 mask as the oracle and satisfies the required sparsity pattern.
