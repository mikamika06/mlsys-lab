## Context

For tiny $x$,

$$
e^{x} - 1 = x + \frac{x^2}{2} + \frac{x^3}{6} + \dots \approx x .
$$

But the obvious code `[math.exp(v) - 1.0 for v in x]` destroys that. In float64, $e^x$ rounds to exactly $1.0$
whenever $|x| < \varepsilon/2 \approx 1.1\times10^{-16}$, so the subtraction returns $0$ and the
relative error is $100\%$. Even at $x = 10^{-9}$ the leading digits cancel and roughly half the
mantissa is lost — this is catastrophic cancellation.

The classic fix (Kahan) rescues the answer *from the rounded exponential itself*. Let

$$
u = \operatorname{fl}(e^{x}), \qquad d = u - 1 .
$$

The subtraction $d$ is exact by Sterbenz's lemma when $u$ is near $1$, but $u$ carries a rounding
error. The correction is to divide by $\log u$ instead of by $x$:

$$
\operatorname{expm1}(x) \;\approx\; d \cdot \frac{x}{\log u} .
$$

The ratio $x / \log u$ is $\approx 1$ and cancels exactly the relative error that $u$ picked up when
it was rounded, because $\log$ maps that error back into the same scale. Two special cases must be
handled separately:

$$
d = 0 \;\Rightarrow\; \text{return } x , \qquad\qquad u = 0 \;\Rightarrow\; \text{return } -1 .
$$

## Task

Implement `exp_minus_one` in `solve.py`:

```python
def exp_minus_one(x: list[float]) -> list[float]:
    ...
```

`x` is a float64 array; return an array of the same shape holding $e^{x} - 1$. The grader evaluates
$x$ over $[-30, 30]$, including a logarithmic sweep down to $10^{-18}$, both signs, exact zero, and
subnormals.

Any library `expm1` (`math.expm1`, …) is blocked: the grader rejects a solution whose
source mentions `expm1`, and additionally raises if `math.expm1` is called during
grading. Use `math.exp` and `math.log`.

Guard the division so no `inf`/`NaN` leaks into the result.

## Example

```python

x = [1e-17, 1e-9, 1.0, -30.0, 0.0]

exp_minus_one(x)
# array([ 1.00000000e-17,  1.00000000e-09,  1.71828183e+00, -1.00000000e+00,  0.0])

[math.exp(v) - 1.0 for v in x]
# array([ 0.00000000e+00,  1.00000008e-09,  1.71828183e+00, -1.00000000e+00,  0.0])
#          ^ all digits lost      ^ ~8 digits lost
```

## What the gate checks

The reference is `math.expm1` evaluated on the same grid (a live Python oracle, nothing hardcoded).

* `rel_err` — the **maximum elementwise** relative error

$$
\max_i \frac{\lvert \hat y_i - y_i \rvert}{\lvert y_i \rvert} \quad\text{over all } y_i \neq 0 ,
$$

  must be $\le 10^{-14}$. A global L2 error would hide the tiny-$x$ failure; this one does not.
The naive `[math.exp(v) - 1.0 for v in x]` scores $1.0$ here.
* `exact_zero_fraction` — where $\operatorname{expm1}(x)$ is exactly $0$, your output must be exactly
  $0$ too; must be $1.0$.

`naive_rel_err` is reported for information.
