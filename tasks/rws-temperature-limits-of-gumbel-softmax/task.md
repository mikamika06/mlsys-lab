## Context

Gumbel-Softmax is a differentiable approximation to sampling from a categorical
distribution. Given logits $z \in \mathbb{R}^k$, fixed Gumbel noise
$g \in \mathbb{R}^k$, and temperature $\tau$, the relaxed sample is

$$
y_i = \frac{\exp((z_i + g_i)/\tau)}
{\sum_{j=1}^{k}\exp((z_j + g_j)/\tau)} .
$$

The temperature controls the limiting behavior. When $\tau \to 0$, the largest
entry of $z+g$ dominates and the output approaches a one-hot vector:

$$
\lim_{\tau \to 0} y = \mathrm{one\_hot}(\arg\max_i(z_i+g_i)).
$$

When $\tau \to \infty$, all logits are suppressed and the output approaches the
uniform distribution:

$$
\lim_{\tau \to \infty} y_i = \frac{1}{k}.
$$

A production implementation must compute the same numerical behavior using a
stable softmax formulation.

## Task

Implement `gumbel_temperature_limits(logits, g, tau_small, tau_large)`:

```python
def gumbel_temperature_limits(
    logits: np.ndarray,
    g: np.ndarray,
    tau_small: float,
    tau_large: float,
) -> tuple[int, np.ndarray]:
    ...
```

The function receives one-dimensional NumPy arrays `logits` and fixed Gumbel
noise `g` of equal length. It must return:

1. `index`: the selected category index in the low-temperature limit, computed
   from `logits + g` as $\tau$ approaches zero.
2. `distribution`: the Gumbel-Softmax distribution computed with `tau_large`,
   which should be close to the high-temperature uniform limit.

Use NumPy operations and a numerically stable softmax. The returned distribution
must be `float64`.

## Example

```python
import numpy as np

logits = np.array([1.0, 2.0, 0.5])
g = np.array([0.1, -0.2, 0.4])

index, distribution = gumbel_temperature_limits(
    logits, g, 1e-6, 1e6
)

# index is the position of the largest value in logits + g
# distribution is close to [1/3, 1/3, 1/3]
```

## What the gate checks

The gate computes the low-temperature oracle using the real operation

$$
\arg\max_i(z_i+g_i)
$$

and checks that the returned index matches it exactly.

It also computes the high-temperature oracle by applying the stable Gumbel-Softmax
formula with the large temperature and compares the returned distribution to the
same NumPy result using relative error. The relative error must satisfy
$\mathrm{rel\_err} \le 10^{-3}$.
