## Context

Activation-aware weight quantization (AWQ) uses a per-input-channel scale to improve the
distribution of weights before quantization. A common transformation rescales weights:

$$
W'_{ij} = W_{ij} s_j ,
$$

where $s_j$ is the scale for input channel $j$.

For a linear layer with input matrix $X$ and weight matrix $W$, the original output is

$$
Y = X W^\top .
$$

The rescaled weights are equivalent only if the inverse scale is folded into the
activations:

$$
Y = (X \oslash s)(W')^\top ,
$$

where $\oslash$ applies element-wise division by the scale vector. Folding the
scale on the wrong side changes the computation because matrix multiplication is
not invariant to arbitrary rescaling of only one operand.

## Task

Implement `fix_awq_scale(W, X, s)`:

```python
def fix_awq_scale(W: list[list[float]], X: list[list[float]], s: list[float]) -> list[list[float]]:
    ...
```

The function receives:

- `W`: a weight matrix of shape $(m, d)$.
- `X`: a batch of activations of shape $(n, d)$.
- `s`: a scale vector of shape $(d,)$.

The function should simulate the corrected AWQ folding and return the linear
layer output:

$$
(X \oslash s)(W \odot s)^\top ,
$$

where $\odot$ multiplies each input channel column of $W$ by the corresponding
scale value. The result must be `float64`.

Use Python operations only.

## Example

```python

W = [[2.0, 3.0], [4.0, 5.0]]
X = [[1.0, 2.0]]
s = [2.0, 4.0]

Y = fix_awq_scale(W, X, s)
# Equivalent to:
# (X / s) @ (W * s).T
```

## What the gate checks

The gate builds a Python oracle by applying the mathematically correct folded
computation and compares the implementation output using the maximum absolute
error:

$$
\max_i |y_i - \hat{y}_i| .
$$

The submitted implementation must match the oracle within the required
tolerance. The gate also uses scale values that make the common bug of folding
the inverse scale into weights or forgetting the activation correction produce
a different result.
