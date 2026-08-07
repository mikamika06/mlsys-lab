## Context

Neural network quantization and inference pipelines often move large activation
outliers into weights by applying a per-channel scale. Given activations
$X \in \mathbb{R}^{m \times d}$ and weights $W \in \mathbb{R}^{d \times n}$,
a diagonal scale matrix $S = \mathrm{diag}(s)$ can be inserted without changing
the product:

$$
XW = (XS)(S^{-1}W).
$$

The scale direction matters. To reduce activation range, the activation tensor
should receive $s$ while the weights receive the inverse scale:

$$
X_{\mathrm{new}} = X \odot s,
\qquad
W_{\mathrm{new}} = s^{-1} \odot W,
$$

where multiplication is broadcast over columns of $X$ and rows of $W$.

Applying the inverse direction,

$$
X \odot s^{-1},
\qquad
s \odot W,
$$

preserves neither the intended migration behavior nor the reduced activation
range.

## Task

Implement `migrate_scale(X, W, s)`:

```python
def migrate_scale(X: list[list[float]], W: list[list[float]], s: list[float]) -> tuple[list[list[float]], list[list[float]]]:
    ...
```

The inputs are:

- `X`: a float list with shape $(m, d)$ containing activations.
- `W`: a float list with shape $(d, n)$ containing weights.
- `s`: a positive float list with shape $(d,)$ containing per-channel scales.

Return `(X_new, W_new)` where:

$$
X_{\mathrm{new}} = X \odot s
$$

and

$$
W_{\mathrm{new}} = s^{-1} \odot W.
$$

The operation must preserve the folded matrix product while reducing the maximum
absolute activation value compared with the incorrectly inverted direction.

## Example

```python

X = [[10.0, 1.0], [8.0, 2.0]]
W = [[0.5, 0.2], [1.0, 3.0]]
s = [0.1, 2.0]

X_new, W_new = migrate_scale(X, W, s)

# X_new contains the activation channels migrated by s:
# [[1.0, 2.0], [0.8, 4.0]]
```

## What the gate checks

The gate computes the expected migration using a Python oracle and compares the
returned tensors with it using the maximum absolute error

$$
\max_i |a_i-b_i|.
$$

The gate also verifies that the fixed direction reduces the activation range
relative to the inverted implementation. The folded product $XW$ is preserved
because the oracle uses the identity

$$
(X \odot s)(s^{-1} \odot W)=XW.
$$
