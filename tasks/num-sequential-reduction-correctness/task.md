## Context

Floating-point addition is not associative. For real numbers,

$$
(a+b)+c = a+(b+c)
$$

but floating-point arithmetic rounds intermediate values, so different evaluation
orders can produce different results.

A sequential left-fold reduction keeps the input order:

$$
s_0 = 0,\qquad s_{i+1} = \operatorname{float64}(s_i + x_i).
$$

A parallel or tree reduction changes the order of additions. Because each
addition may round, the final value can differ even when all operations use the
same floating-point format.

## Task

Implement `sequential_sum(values)`:

```python
def sequential_sum(values: list[float]) -> float:
    ...
```

The function receives a list of floats of floating-point values and
must return the sum produced by a left fold in the original input order.
Accumulate using Python `float64` arithmetic and return the scalar result.

Do not use built-in reduction functions or other reduction operations, because their reduction order
may differ from the required sequential order.

## Example

```python

x = [1.0, 2.0, 3.0]
result = sequential_sum(x)
# 6.0
```

## What the gate checks

The gate builds a reference result by executing the left-fold algorithm directly
and compares the candidate output with relative error:

$$
\mathrm{rel\_err} =
\frac{|y_{\mathrm{candidate}}-y_{\mathrm{reference}}|}
{|y_{\mathrm{reference}}|+10^{-12}} .
$$

The maximum relative error across several floating-point cancellation cases must
satisfy $\mathrm{rel\_err} \le 10^{-9}$.
