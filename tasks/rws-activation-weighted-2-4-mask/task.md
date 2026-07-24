## Context

Structured sparsity formats can require keeping exactly two weights from every group of four weights. For a weight matrix $W \in \mathbb{R}^{m \times n}$, a 2:4 mask $M$ keeps two entries in each consecutive group of four columns for every row.

A magnitude-only pruning rule selects entries using $|W_{ij}|$. This ignores the input activations used by the matrix multiplication.

For an activation matrix $X \in \mathbb{R}^{n \times k}$, define the scale of input column $j$ as

$$
s_j = \lVert X_j \rVert_2 .
$$

The activation-weighted importance score for a weight is

$$
I_{ij} = |W_{ij}|s_j .
$$

Selecting the two largest scores in each group of four columns gives a mask that is adapted to the observed activation distribution.

The reconstruction error is measured after pruning:

$$
E = \lVert WX - (W \odot M)X \rVert_F^2 .
$$

## Task

Implement:

```python
def activation_weighted_2_4_mask(W: np.ndarray, X: np.ndarray):
    ...
```

The function receives:

- `W`: a floating point matrix of shape $(m,n)$.
- `X`: a floating point matrix of shape $(n,k)$.

The number of columns of `W` is always divisible by four.

Return:

```python
(mask, error)
```

where:

- `mask` is an integer NumPy array with the same shape as `W`.
- Each consecutive block of four columns in every row contains exactly two ones.
- `error` is the floating point value of

$$
\lVert WX - (W \odot M)X \rVert_F^2 .
$$

The mask must be selected using $|W_{ij}|\lVert X_j\rVert_2$.

## Example

```python
import numpy as np

W = np.array([[3., 1., 5., 2.]])
X = np.array([[1., 0.],
              [2., 0.],
              [0., 1.],
              [0., 3.]])

mask, error = activation_weighted_2_4_mask(W, X)

# scores are [3, 2, 5, 6]
# selected columns are 2 and 3
# mask is [[0, 0, 1, 1]]
```

## What the gate checks

The grader builds the expected mask with a NumPy oracle using the activation-weighted scores. It checks that the returned mask and reconstruction error match the oracle.

It also computes a plain magnitude 2:4 mask using $|W|$ and verifies that the activation-weighted result is no worse on the measured reconstruction error:

$$
E_{\mathrm{weighted}} \leq E_{\mathrm{magnitude}} .
$$

A solution that ignores $X$ and only uses weight magnitude will fail cases where activation scales change which weights are most important.
