## Context

For a stream of values $x_1, x_2, \dots, x_n$, the first four central moments describe the shape of the distribution. The online algorithm keeps running quantities instead of storing all samples:

$$
M_k = \sum_{i=1}^{n}(x_i-\mu)^k .
$$

Welford's method updates the mean and the higher order moments as each new value arrives. This avoids the numerical instability of repeatedly computing powers around a large mean.

After processing all samples, population skewness is

$$
\gamma_1 = \frac{M_3 / n}{(M_2 / n)^{3/2}},
$$

and excess kurtosis is

$$
\gamma_2 = \frac{M_4 / n}{(M_2 / n)^2} - 3 .
$$

The implementation must use running updates for $M_2$, $M_3$, and $M_4$ rather than storing the input and recomputing all moments at the end.

## Task

Implement `online_moments(values)`:

```python
def online_moments(values):
    ...
```

The function receives an iterable of real-valued samples and returns a tuple:

```python
(skewness, excess_kurtosis)
```

Use a numerically stable online update of the mean and accumulated moments $M_2$, $M_3$, and $M_4$. The returned values must be Python floats.

The function should return `(0.0, 0.0)` for an empty input or for an input with zero variance.

## Example

```python
values = [1.0, 2.0, 3.0, 4.0]
skew, kurt = online_moments(values)

# skew is approximately 0.0
# excess kurtosis is approximately -1.36
```

## What the gate checks

The gate computes the reference skewness and excess kurtosis from a NumPy batch calculation over several generated inputs. The returned pair is compared using the relative error metric:

$$
\mathrm{rel\_err} =
\frac{\lVert y_{\mathrm{candidate}}-y_{\mathrm{reference}}\rVert}
{\lVert y_{\mathrm{reference}}\rVert + 10^{-12}} .
$$

A value below $10^{-8}$ is required. Implementations that only track mean and variance or that compute unstable higher moments will fail.
