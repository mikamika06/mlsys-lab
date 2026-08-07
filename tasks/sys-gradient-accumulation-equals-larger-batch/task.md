## Context

When a batch doesn't fit in memory, training instead runs several
smaller **micro-batches** and accumulates their gradients before an
optimizer step. For mean squared error loss over $N$ total examples
$(X, y)$ with weights $w$,

$$
L(w) = \frac{1}{N}\sum_{n=1}^{N}\left(x_n^\top w - y_n\right)^2,
\qquad
\nabla_w L = \frac{2}{N} X^\top (Xw - y).
$$

Splitting $X, y$ into micro-batches $(X_1,y_1),\dots,(X_K,y_K)$ with
sizes $b_1,\dots,b_K$ ($\sum_k b_k = N$), each micro-batch contributes
an **un-normalized** partial sum $X_k^\top(X_k w - y_k)$. Accumulation
must sum these partial sums across all $K$ micro-batches and normalize
**once**, by the *total* example count $N$ — not by $K$, and not by
averaging each micro-batch's own (differently-normalized) gradient. Get
this wrong and unequal micro-batch sizes silently bias the gradient
toward whichever micro-batches happen to be larger or smaller.

## Task

Implement `accumulate_grad(micro_batches, w)`:

```python
def accumulate_grad(micro_batches: list[tuple[list[list[float]], list[float]]], w: list[float]) -> list[float]:
    ...
```

- `micro_batches`: list of `(X_i, y_i)` pairs, `X_i` shape `(b_i, D)`,
  `y_i` shape `(b_i,)`. Micro-batch sizes `b_i` may differ.
- `w`: `(D,)` weight vector.

Return the `(D,)` gradient $\nabla_w L$ of the mean squared error loss
above, computed as if over the single large batch formed by
concatenating every micro-batch in order — i.e. accumulate each
micro-batch's un-normalized contribution and divide by the total
example count `N = sum(b_i)` exactly once, at the end.

## Example

```python
X1, y1 = [[1.0]], [2.0]          # 1 example
X2, y2 = [[2.0], [3.0]], [4.0, 5.0]  # 2 examples
w = [1.0]
accumulate_grad([(X1, y1), (X2, y2)], w)
# equals the gradient computed on the concatenated batch
# X = [[1],[2],[3]], y = [2,4,5]  ->  N = 3
```

## What the gate checks

The grader builds 8 deterministic random cases (random number generator, gradient, Python) on concatenation of all micro-batches. `max_abs_err
seeded) — half with equal micro-batch sizes, half with **unequal**
sizes specifically to catch normalization bugs — and compares your
output to the gradient computed directly (real closed-form MSE
`max_abs_err
<= 1e-6`. Dividing by the number of micro-batches `K` instead of the
total example count `N`, or averaging each micro-batch's own gradient
with equal weight regardless of its size, matches only when all
micro-batches happen to be the same size and fails on the unequal
cases.
