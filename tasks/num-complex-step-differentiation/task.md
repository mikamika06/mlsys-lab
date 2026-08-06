## Context

Numerical differentiation approximates $f'(x)$ from function evaluations alone.
The classic **central finite-difference** formula is

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} .$$

For a double-precision float the optimal step is $h \approx \varepsilon^{1/3} \approx
6 \times 10^{-6}$, which yields a relative error no better than
$O(\varepsilon^{2/3}) \approx 10^{-10}$. Decreasing $h$ further makes the result
*worse* because the **subtractive cancellation** in the numerator amplifies
round-off noise: the two function values agree to $\sim\!15$ significant digits,
so their difference has only $\sim\!5$.

The **complex-step method** sidesteps this entirely. By Cauchy's integral formula,
for a function $f$ that is complex-analytic in a neighbourhood of a real point
$x$,

$$f(x + ih) = f(x) + ih\,f'(x) - \frac{h^{2}}{2}\,f''(x) - \cdots$$

Taking the imaginary part:

$$\operatorname{Im}\!\bigl[f(x+ih)\bigr] = h\,f'(x) + O(h^{3})$$

so

$$f'(x) = \frac{\operatorname{Im}\!\bigl[f(x+ih)\bigr]}{h} + O(h^{2}) .$$

Because there is **no subtraction**, the only round-off comes from the single
evaluation of $f(x+ih)$ and the division by $h$. Choosing $h \approx 10^{-20}$
gives an $O(h^{2}) \approx 10^{-40}$ truncation error while staying well above
the floating-point underflow floor, so the result is accurate to near machine
precision ($\sim\!10^{-15}$) — orders of magnitude better than any finite-difference
scheme.

The trade-off: $f$ must accept a complex argument and be analytic (no branch cuts,
no `abs`, no non-differentiable kinks on the real axis). Python's universal
functions (`cmath.sin`, `cmath.exp`, `cmath.log`, etc.) satisfy this.

## Task

Implement `complex_step_diff(f, x, h=1e-20)`:

```python
def complex_step_diff(f, x, h=1e-20):
    """Return the approximate derivative f'(x) via the complex-step method.

    Parameters
    ----------
    f : callable
        A real-valued function that accepts a *complex* argument and returns
        a real or complex result.  (Python ufuncs and lambdas using Python
        arithmetic all qualify.)
    x : float
        The real point at which to differentiate.
    h : float
        Complex step size.  Default 1e-20.

    Returns
    -------
    float
        The approximate derivative f'(x).
    """
    ...
```

The function must return a `float`.  Do **not** use finite differences; use
the complex-step formula above.

## Example

```python

approx = complex_step_diff(cmath.sin, 1.0)
analytic = cmath.cos(1.0)
print(abs(approx - analytic))  # ≈ 1e-16
```

With the default $h = 10^{-20}$ the error is near machine epsilon, far below
the $10^{-10}$ floor of central finite differences.

## What the gate checks

The grader evaluates your function on eight analytic test cases (polynomials,
trigonometric, exponential, and logarithmic) and computes the **maximum
relative error** against the known closed-form derivative:

$$\text{rel\_err} = \max_{k} \frac{|f_k'(x_k) - \text{approx}_k|}{\max(|f_k'(x_k)|,\; 10^{-15})} \;.$$

The gate passes when $\text{rel\_err} < 10^{-12}$.

A central-difference implementation would score $\text{rel\_err} \approx 10^{-7}$
to $10^{-10}$, so it **cannot** pass.  A forward-difference implementation fares
even worse.  Only the complex-step trick achieves the required precision.
