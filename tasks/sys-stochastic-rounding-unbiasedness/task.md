## Context

Low-precision arithmetic reduces storage and computation costs by representing values with fewer bits. Quantization introduces rounding error because many real numbers do not have an exact representation in a smaller format.

For a value $x$ between two adjacent representable values $l$ and $u$, stochastic rounding chooses one of the two neighbors randomly:

$$
R(x)=
\begin{cases}
l & \text{with probability } 1-p,\\
u & \text{with probability } p,
\end{cases}
$$

where

$$
p=\frac{x-l}{u-l}.
$$

The expected rounded value is

$$
\mathbb{E}[R(x)] = (1-p)l+pu=x.
$$

This makes stochastic rounding unbiased. When many independently rounded samples are averaged, the result should approach the original value rather than drifting consistently upward or downward.

## Task

Implement `stochastic_round(x, rng)`:

```python
def stochastic_round(x: np.ndarray, rng) -> np.ndarray:
    ...
```

The function receives a NumPy `float32` array and a NumPy random generator. Return a `float32` array containing stochastic rounding of every element to the nearest `float16` representable values.

For each element, find the adjacent lower and upper `float16` values. Sample the upper value with probability

$$
p=\frac{x-l}{u-l}
$$

and the lower value otherwise. If the input value is already exactly representable as `float16`, return that value. Use the supplied `rng` object for all random numbers.

## Example

```python
import numpy as np

x = np.array([1.0, 1.0005], dtype=np.float32)
rng = np.random.default_rng(0)

y = stochastic_round(x, rng)

# y contains only float16-representable values.
```

## What the gate checks

The gate computes a reference distribution using the same stochastic rounding algorithm implemented from NumPy `float16` conversion rules. It then averages many samples from the candidate implementation and compares the candidate mean with the oracle mean using

$$
\mathrm{rel\_err}=
\frac{\lVert \mathrm{mean}(R(x))-\mathrm{mean}(R_{\mathrm{oracle}}(x))\rVert_2}
{\lVert \mathrm{mean}(R_{\mathrm{oracle}}(x))\rVert_2+10^{-12}}.
$$

The score must satisfy $\mathrm{rel\_err}\leq10^{-3}$. A deterministic `float16` cast fails because it always selects one neighbor instead of sampling according to the unbiased probability.
