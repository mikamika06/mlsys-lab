## Context

Matrix multiplication combines rows and columns through a dot product. For two
matrices $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$,
the output $C \in \mathbb{R}^{m \times n}$ is defined by

$$
C_{ij} = \sum_{t=1}^{k} A_{it} B_{tj}.
$$

A batch of matrices adds a leading dimension. Given
$A \in \mathbb{R}^{b \times m \times k}$ and
$B \in \mathbb{R}^{b \times k \times n}$, each batch element contains an
independent matrix multiplication:

$$
C_{sij} = \sum_{t=1}^{k} A_{sit} B_{stj}.
$$

This operation appears frequently in numerical computing and machine learning.
A direct implementation can be built by iterating over the batch dimension and
then over the output rows, columns, and reduction dimension.

## Task

Implement `batched_matmul(A, B)`:

```python
def batched_matmul(A: list, B: list) -> list:
    ...
```

The function receives two 3-D list with shapes $(b, m, k)$ and
$(b, k, n)$ and returns a 3-D array with shape $(b, m, n)$ containing the
batched matrix product.

Implement the multiplication from scratch using explicit loops over the batch,
output rows, output columns, and shared dimension. Do not call `matmul`,
`dot`, or equivalent matrix multiplication helpers.

The returned array should contain floating point values.

## Example

```python

A = [
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
]

B = [
    [[1, 0], [0, 1]],
    [[2, 1], [1, 2]]
]

C = batched_matmul(A, B)

# [
#   [[1., 2.], [3., 4.]],
#   [[16., 17.], [22., 23.]]
# ]
```

## What the gate checks

The gate computes a reference result using Python's `matmul` implementation and
compares the submitted function output with the oracle result.

The maximum absolute error

$$
\max_{i,j,k} |C_{ijk}^{\mathrm{candidate}} - C_{ijk}^{\mathrm{reference}}|
$$

must be less than $10^{-6}$.
