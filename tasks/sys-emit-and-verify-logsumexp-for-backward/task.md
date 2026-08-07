## Context

In fused attention kernels such as FlashAttention, the forward pass often emits a
per-row logsumexp (LSE) value that is reused by the backward pass. For a score
matrix $S \in \mathbb{R}^{n \times m}$, each row has

$$
\mathrm{LSE}_i = \log\left(\sum_{j=1}^{m} e^{S_{ij}}\right).
$$

The softmax probabilities can then be reconstructed without storing the full
probability matrix:

$$
P_{ij} = e^{S_{ij} - \mathrm{LSE}_i}.
$$

Computing the denominator in log space avoids unnecessary overflow and is the
same idea used by numerically stable attention implementations.

## Task

Implement `emit_lse(S)`:

```python
def emit_lse(S: list[list[float]]) -> list[float]:
    ...
```

The function receives a list of lists of floats of attention scores with shape
$(n, m)$ and returns a 1-D `float64` list containing the LSE value for
each row.

The implementation must compute the row-wise logsumexp values accurately. Do not
return the softmax matrix. The output length must equal the number of rows in
`S`.

## Example

```python

S = [[1.0, 2.0, 3.0],
              [0.0, 0.0, 0.0]]

lse = emit_lse(S)

# exp(S - lse[:, None]) reconstructs:
# [[0.09003057, 0.24472847, 0.66524096],
#  [0.33333333, 0.33333333, 0.33333333]]
```

## What the gate checks

The gate computes a Python oracle for row logsumexp and uses the returned LSE
values to reconstruct softmax probabilities with

$$
\hat{P}_{ij}=e^{S_{ij}-\mathrm{LSE}_i}.
$$

The reconstructed probabilities are compared with a dense Python softmax oracle.
The reported metric is `max_abs_err`, and it must be less than $10^{-6}$.
