## Context

AWQ (Activation-aware Weight Quantization) reduces quantization error on
"salient" input channels by **migrating** a per-input-channel scale $s_j >
0$ from the activations onto the weights before quantizing. For a linear
layer $y = X W$ with $X \in \mathbb{R}^{b \times d_{\text{in}}}$,
$W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$, and a scale vector
$s \in \mathbb{R}^{d_{\text{in}}}$, the transform must leave the *product*
unchanged — the whole point is to make $W$'s salient rows larger (so they
survive quantization better) while shrinking the matching activation column
by the exact same factor, so the math is invariant:

$$
X'_{:,j} = \frac{X_{:,j}}{s_j}, \qquad W'_{j,:} = s_j \cdot W_{j,:}
\qquad\Longrightarrow\qquad X' W' = X W
$$

A buggy implementation was submitted that instead multiplies **both** sides
by $s$:

```python
X_prime = X * s          # WRONG — should divide
W_prime = W * s[:, None]
```

Multiplying $X$ by $s$ instead of dividing breaks the invariant: each column
$j$ of the product picks up an extra factor of $s_j$ that is never undone,
so $X' W' \neq X W$ in general (the two only happen to agree when $s_j = 1$
for all $j$).

## Task

Fix `apply_migration_scale` so the invariant holds:

```python
def apply_migration_scale(X: list[list[float]], W: list[list[float]], s: list[float]) -> tuple:
    ...
```

* `X` — activations, shape $(b, d_{\text{in}})$.
* `W` — weights, shape $(d_{\text{in}}, d_{\text{out}})$.
* `s` — positive per-input-channel scale, shape $(d_{\text{in}},)$.

Return `(X_prime, W_prime)` where `X_prime = X / s` (broadcast over columns)
and `W_prime = W * s[:, None]` (broadcast over rows), so that
`X_prime @ W_prime` reproduces `X @ W` up to floating-point rounding.

## Example

```python
X = [[2.0, 4.0]]
W = [[1.0], [1.0]]
s = [2.0, 0.5]

X_prime, W_prime = apply_migration_scale(X, W, s)
# X_prime = [[1.0, 8.0]]      (X / s)
# W_prime = [[2.0], [0.5]]    (W * s[:, None])
allclose(X_prime @ W_prime, X @ W) # -> True ( [[6.0]] == [[6.0]] )
```

## What the gate checks

A single **max_abs_err** gate builds several random `(X, W, s)` triples,
computes `X @ W` directly, reconstructs `X_prime @ W_prime` from your
returned tuple, and requires the maximum absolute difference between the
two to be below `1e-6`. The buggy "multiply both sides" version fails this
because the invariant no longer holds for `s != 1`. Any exception or wrong
return shape also fails the gate.
