## Context

At inference time, BatchNorm is just a fixed per-channel affine transform
(using its frozen running statistics, not batch statistics) — so a
`Linear -> BatchNorm` pair can always be collapsed into a single
equivalent `Linear` layer. This is one of the most common graph fusions
production compilers (TensorRT, ONNX Runtime, torch.compile) perform,
because it removes an entire op's worth of memory traffic for free, with
**zero** change in the math.

For a linear layer $y = Wx + b$ followed by
$\mathrm{BN}(y) = \gamma \cdot \dfrac{y - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$
(per output channel, using running mean $\mu$ and running variance
$\sigma^2$), define the per-channel scale
$s = \gamma / \sqrt{\sigma^2 + \epsilon}$. Then:

$$
W' = s \odot W \quad \text{(scaling each output row } i \text{ by } s_i\text{)}, \qquad
b' = s \odot (b - \mu) + \beta
$$

and $W'x + b' \equiv \mathrm{BN}(Wx + b)$ for every input $x$ — exactly,
not approximately.

## Task

Implement `fold_bn_into_linear`:

```python
def fold_bn_into_linear(W: list[list[float]], b: list[float], gamma: list[float], beta: list[float], running_mean: list[float], running_var: list[float], eps: float) -> tuple[list[list[float]], list[float]]:
    ...
```

- `W`: `(out_features, in_features)` float64 weight matrix.
- `b`: `(out_features,)` float64 bias.
- `gamma`, `beta`, `running_mean`, `running_var`: `(out_features,)`
  float64 BatchNorm parameters (one value per output channel).
- `eps`: float, BatchNorm's numerical-stability constant.

Return `(W_folded, b_folded)`, the single equivalent linear layer's
weight and bias, per the formulas above.

## Example

```python

W = [[1.0, 2.0], [3.0, 4.0]]
b = [0.5, -0.5]
gamma = [2.0, 0.5]
beta = [0.0, 1.0]
running_mean = [1.0, 0.0]
running_var = [3.0, 8.0]

W_folded, b_folded = fold_bn_into_linear(W, b, gamma, beta, running_mean, running_var, eps=1e-5)
# scale s = gamma / sqrt(running_var + eps)
# W_folded[i] = s[i] * W[i]; b_folded[i] = s[i]*(b[i]-running_mean[i]) + beta[i]
# For any x: W_folded @ x + b_folded == BN(W @ x + b) (BN using running stats).
```

## What the gate checks

The grader builds several `(W, b, gamma, beta, running_mean, running_var,
eps)` scenarios from a seeded Python generator (varying shapes, some
`gamma` values near zero, and both positive and negative `beta`/`b`) and
computes the reference `W_folded`, `b_folded` independently in Python from
the formulas above, never calling your function. It also feeds several
random input vectors `x` through both `BN(Wx + b)` (using the *unfolded*
parameters and BatchNorm's own formula) and `W_folded @ x + b_folded`
(using your folded parameters), as an end-to-end equivalence check.

`max_abs_err` is the worst-case elementwise absolute error across your
`W_folded`, your `b_folded`, and every probed output vector, over every
scenario, and the gate requires `<= 1e-6`. Forgetting to subtract
`running_mean` before scaling, applying the scale to `b` before
subtracting the mean, or scaling `W` by `gamma` alone (without dividing
by $\sqrt{\sigma^2+\epsilon}$) will all produce a folded layer whose
output disagrees with `BN(Wx + b)`.
