## Context

The log-sum-exp function

$$\mathrm{LSE}(x) = \log\left(\sum_{i=1}^{n} e^{x_i}\right)$$

shows up everywhere probabilities are combined in log-space (softmax
normalizers, mixture-model log-likelihoods, cross-entropy). Computed
literally — `log(sum(exp(x)))` — it breaks in float64 the moment $x$
strays outside a narrow range:

* If any $x_i \gtrsim 709$, `exp(x_i)` overflows to `inf`, and
  `log(inf) = inf` — even though the true value of $\mathrm{LSE}(x)$ is
  perfectly finite (roughly $\max_i x_i$).
* If every $x_i \lesssim -745$, `exp(x_i)` underflows to `0.0` for all
  $i$, and `log(0) = -inf` — again, even though the true value is finite.

The standard fix shifts by the maximum entry before exponentiating:

$$\mathrm{LSE}(x) = m + \log\left(\sum_{i=1}^{n} e^{x_i - m}\right), \qquad m = \max_i x_i.$$

Now every exponent $x_i - m \le 0$, so $e^{x_i-m} \in (0, 1]$ — it can never
overflow, and it underflows to $0$ only for terms that are negligible next
to the largest one anyway, which does not change the (correctly-rounded)
result.

## Task

Implement `logsumexp_stable`:

```python
def logsumexp_stable(x: list[float]) -> float:
    ...
```

- `x` — list of floats of floats, possibly containing very large positive or very large negative entries.


Return $\mathrm{LSE}(x)$ as a plain Python `float`, using the max-shift trick so the result stays finite and accurate no matter how extreme the entries of `x` are.

## Example

```python
x = [1000.0, 1000.5, 999.0]
print(logsumexp_stable(x))   # ~1001.157..., NOT inf

x2 = [-1000.0, -1000.5, -999.0]
print(logsumexp_stable(x2))  # ~-998.536..., NOT -inf
```

The naive `float(math.log(sum(math.exp(v) for v in x)))` returns `inf` on the first example and `-inf` on the second.

## What the gate checks

A single gate named **rel_err** compares your output against a real high-precision oracle: the same sum-of-exponentials computed with `mpmath` at 50 decimal digits of precision (immune to float64 overflow/underflow), then converted back to a `float`. It runs on the overflow and underflow fixtures, a mixed-extreme-range case, and several ordinary random lists. The threshold is $10^{-13}$ — a correct stable implementation matches the oracle to within float64 rounding error on every case, while the naive formula returns a non-finite value on the extreme cases and is scored as failing.
