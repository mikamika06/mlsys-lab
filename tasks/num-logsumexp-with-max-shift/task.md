## Context

The log-sum-exp function

$$
\mathrm{LSE}(x) = \log \sum_{i=1}^{n} e^{x_i}
$$

shows up everywhere in log-domain arithmetic (softmax, cross-entropy,
mixture-model log-likelihoods). Computed naively it overflows: if any
$x_i$ is a few hundred or more, $e^{x_i}$ already exceeds the range of a
`float64`.

The fix is the **max-shift trick**. Let $m = \max_i x_i$. Then

$$
\mathrm{LSE}(x) = m + \log \sum_{i=1}^{n} e^{x_i - m}.
$$

This is mathematically identical (factor $e^m$ out of the sum, then take
the log back out), but now every exponent $x_i - m \le 0$, so
$e^{x_i - m} \in (0, 1]$ and nothing overflows — and since at least one
term equals exactly $e^0 = 1$, the sum can't underflow to zero either.

## Task

Implement `logsumexp`:

```python
def logsumexp(x: list[list[float]], axis: int=-1) -> list[float]:
    ...
```

* `x` — a list of `float64` values, which may range anywhere from
  $-10^4$ to $10^4$.
* `axis` — the axis to reduce over, with the same semantics as
`sum(..., axis=axis)` and `keepdims=False`.

Use the max-shift trick above — do not exponentiate the raw values.

## Example

```python
x = [1e4, 1e4 + 1.0, 1e4 - 3.0]
print(logsumexp(x))
# -> 10001.31326...   (== 1e4 + log(1 + e + e^-3)), NOT inf
```

A naive `math.log(sum(math.exp(x)))` on this input overflows to `inf`.

## What the gate checks

The grader compares your output against `scipy.special.logsumexp` on 8
random arrays with entries drawn from $[-10^4, 10^4]$, reduced along a
random axis each time.

* **rel_err** — mean relative L2 error against the `scipy` reference.
  Must be below `1e-12` (i.e. essentially float64-exact — this is a
  closed-form numeric identity, not an approximation).
