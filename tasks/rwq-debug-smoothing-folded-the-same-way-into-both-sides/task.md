## Context

Some optimization transforms insert a diagonal smoothing matrix between two
factors. Let $X \in \mathbb{R}^{m \times d}$ and $W \in \mathbb{R}^{d \times n}$.
A smoothing vector $s \in \mathbb{R}^d$ defines the diagonal matrix

$$S = \mathrm{diag}(s).$$

A common goal is to keep the product unchanged while redistributing scale:

$$XW = (XS)(S^{-1}W).$$

The inverse scaling is important. If both sides are multiplied by $S$, the
scaling is applied twice:

$$ (XS)(SW) = X S^2 W, $$

which generally changes the result.

## Task

Implement `fold_smoothing(X, W, s)`:

```python
def fold_smoothing(X: list[list[float]], W: list[list[float]], s: list[float]) -> tuple[list[list[float]], list[list[float]]]:
    ...
```

The function receives matrices $X$ with shape $(m, d)$, $W$ with shape
$(d, n)$, and a smoothing vector $s$ with length $d$.

Return transformed matrices `X_new` and `W_new` such that:

$$X_{\mathrm{new}} = X S$$

and

$$W_{\mathrm{new}} = S^{-1} W.$$

The returned arrays must be list with floating point values.

## Example

```python

X = [[2.0, 4.0]]
W = [[3.0], [5.0]]
s = [2.0, 4.0]

X_new, W_new = fold_smoothing(X, W, s)

# X_new == [[4.0, 16.0]]
# W_new == [[1.5], [1.25]]

# X_new @ W_new == X @ W
```

## What the gate checks

The gate generates deterministic matrices and compares the submitted transform
with a Python oracle implementation of the smoothing fold. The maximum absolute
difference between the candidate product and the original product,

$$\max |X_{\mathrm{new}}W_{\mathrm{new}} - XW|,$$

must be at most $10^{-5}$.

A transform that multiplies both $X$ and $W$ by $s$ fails because it produces
$XS^2W$ instead of preserving $XW$.
