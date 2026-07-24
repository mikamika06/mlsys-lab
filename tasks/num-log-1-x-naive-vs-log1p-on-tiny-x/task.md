## Context

The natural logarithm satisfies $\log(1+x)=x-\frac{x^2}{2}+\dots$ for small $x$. Computing it as $\log(1+x)$ with a floating‑point `np.log` suffers from catastrophic cancellation when $x\ll 1$, because the argument to `log` is close to one and the subtraction in the logarithm’s series expansion loses precision. NumPy offers a dedicated routine `np.log1p(x)` that evaluates $\log(1+x)$ accurately for tiny arguments by using a special algorithm.

## Task

Implement `log1p_tiny(x: np.ndarray) -> np.ndarray`:

```python
def log1p_tiny(x: np.ndarray) -> np.ndarray:
    ...
```

It receives a 1‑D NumPy array of type `float64` containing values in the range $[10^{-16},\,10^{-8}]$ and must return an array of the same shape with the accurate logarithm $\log(1+x)$ computed as a float64. The implementation should be fully vectorised; no Python loops are allowed.

## Example

```python
import numpy as np
x = np.array([1e-12, 5e-9])
y = log1p_tiny(x)
# y ≈ [-2.302585092994046e-12, -4.999987500000156e-09]
```

## What the gate checks

The grader computes a reference array using `np.log1p` and evaluates the global relative L² error

$$\mathrm{rel\_err} = \frac{\lVert y_{\text{ref}}-y_{\text{cand}}\rVert}{\lVert y_{\text{ref}}\rVert}.$$

The candidate passes if $\mathrm{rel\_err}\le 10^{-14}$ on a set of ten random tiny values. A naive implementation that uses `np.log(1+x)` will fail this gate because of catastrophic cancellation.
