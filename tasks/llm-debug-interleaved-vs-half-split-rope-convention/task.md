## Context

Rotary position embedding (RoPE) rotates pairs of hidden dimensions by a
position-dependent angle. For a vector $x \in \mathbb{R}^{d}$ with even
dimension $d$, the GPT-NeoX style convention splits the vector into two halves.

For each pair of dimensions $(i, i+d/2)$, the rotation is

$$
\begin{aligned}
y_i &= x_i \cos(\theta_i) - x_{i+d/2}\sin(\theta_i),\\
y_{i+d/2} &= x_i \sin(\theta_i) + x_{i+d/2}\cos(\theta_i).
\end{aligned}
$$

The angle for dimension pair $i$ at position $p$ is

$$
\theta_i = p \cdot 10000^{-2i/d}.
$$

A common implementation bug uses adjacent pairs $(0,1), (2,3), \dots$
instead of the half-split pairs. Both versions produce valid rotations, but
they represent different models and are not interchangeable.

## Task

Implement `apply_rope(x, position)`:

```python
def apply_rope(x: list[float], position: int) -> list[float]:
    ...
```

The input `x` is a list of floats with an even number of
dimensions. Return a new `float64` list containing the RoPE-transformed
vector using the GPT-NeoX half-split convention.

The input vector must not be modified.

## Example

```python

x = [1., 2., 3., 4.]
y = apply_rope(x, 5)
```

For dimension $d=4$, the first pair is dimensions $(0,2)$ and the second pair
is dimensions $(1,3)$ because the split point is $d/2=2$.

## What the gate checks

The gate builds its expected outputs by computing the half-split RoPE formula
directly with Python. The returned vector is compared with the oracle result
using the maximum absolute error

$$
\max_i |y_i - \hat{y}_i|.
$$

The error must be less than $10^{-6}$. An implementation that rotates adjacent
dimensions instead of half-split dimensions will fail on nontrivial vectors.
