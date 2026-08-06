## Context

Before you trust a hand-written backward pass you compare it against finite
differences. The one-sided difference

$$D_+ f(x) = \frac{f(x+h) - f(x)}{h} = f'(x) + \frac{h}{2}f''(x) + O(h^2)$$

is only first-order accurate. The **central** difference cancels the even Taylor
terms:

$$D_c f(x) = \frac{f(x+h) - f(x-h)}{2h} = f'(x) + \frac{h^2}{6}f'''(x) + O(h^4),$$

so its truncation error is $O(h^2)$ instead of $O(h)$. Against it works
floating-point cancellation: subtracting two nearly equal values loses
$\approx \varepsilon\,\vert{}f(x)\vert{}/h$ of accuracy. The total error is therefore

$$\underbrace{\tfrac{h^2}{6}\vert{}f'''\vert{}}_{\text{truncation}} + \underbrace{\tfrac{\varepsilon \vert{}f\vert{}}{h}}_{\text{roundoff}},$$

minimised near $h \approx \varepsilon^{1/3} \approx 6\times10^{-6}$ — which is why
$h = 10^{-5}$ is the classic default for a central-difference gradient check in
float64.

To report the disagreement scale-independently, gradient checkers use a
symmetric relative difference rather than a raw subtraction:

$$r = \frac{\vert{}D_c f(x) - g(x)\vert{}}{\max\bigl(\vert{}D_c f(x)\vert{} + \vert{}g(x)\vert{},\ 10^{-12}\bigr)},$$

where $g$ is the analytic gradient under test. The floor in the denominator keeps
$r$ finite when both gradients are zero.

## Task

Implement two functions.

```python
def central_diff(f: Callable[[float], float], x: float, h: float = 1e-5) -> float: ...
def grad_check(f, grad_f, x: float, h: float = 1e-5) -> float: ...
```

- `central_diff(f, x, h)` returns $\dfrac{f(x+h) - f(x-h)}{2h}$ as a Python `float`. `f` is any callable mapping a scalar to a scalar.
- `grad_check(f, grad_f, x, h)` returns the symmetric relative difference $r$ defined above, using the central difference of `f` and the value `grad_f(x)`. Use exactly the $\max(\cdot,\ 10^{-12})$ floor in the denominator.


Both must work for arbitrary callables — do not special-case any particular `f`.

## Example

```python
import math

f  = lambda x: math.sin(x) * math.exp(-0.3 * x)
df = lambda x: math.exp(-0.3 * x) * (math.cos(x) - 0.3 * math.sin(x))

print(central_diff(f, 0.7, 1e-5))
# 0.4459741460372247        (analytic: 0.4459741466...)

print(grad_check(f, df, 0.7))
# 6.3e-11                   -> gradient agrees

bad = lambda x: 1.5 * df(x)
print(grad_check(f, bad, 0.7))
# 0.2                       -> gradient is wrong
```

## Example of what breaks

A forward difference `(f(x + h) - f(x)) / h` is off by roughly $\tfrac{h}{2}|f''(x)| \approx 5\times10^{-6}$ relative — a thousand times over the gate. So is a central difference divided by `h` instead of `2h`.

## What the gate checks

The grader evaluates six analytic functions (polynomial, damped sine, `log1p`-like, `tanh`, sigmoid, rational) at 12 points drawn from a seeded random generator, with $h = 10^{-5}$.

- `rel_err` — your `central_diff` values stacked into one list versus the **closed-form analytic derivatives** computed by the grader, scored with the global relative $L_2$ error $\lVert \hat g - g\rVert_2 / \lVert g \rVert_2$. Must be $\le 10^{-6}$; a correct central difference lands near $10^{-11}$.
- `check_err` — the largest absolute deviation of your `grad_check` from the grader's own evaluation of the formula above, over both correct and deliberately corrupted analytic gradients. Must be $\le 10^{-9}$.
- `detect_ok` — must be `1.0`: `grad_check` reports $\le 10^{-6}$ for every correct gradient and $> 10^{-3}$ for every corrupted one.
