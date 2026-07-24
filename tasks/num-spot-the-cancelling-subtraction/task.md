## Context

When two nearly equal numbers are subtracted, the significant digits of the result can be lost.  
Let  

$$x_{\text{float}} = a - b \quad\text{(computed in IEEE‑754 double precision)}$$  

and  

$$x_{\text{exact}} = \operatorname{Decimal}(a) - \operatorname{Decimal}(b)$$  

be the exact difference computed with arbitrary precision.  
The relative error is

$$
\mathrm{rel\_err} = \frac{|\,x_{\text{float}}-x_{\text{exact}}\,|}{|\,x_{\text{exact}}\,|}\, .
$$

If $\mathrm{rel\_err}$ exceeds $10^{-12}$, the subtraction is said to suffer from catastrophic cancellation.  
Additionally, if the exact difference is zero but the floating‑point result is non‑zero (due to rounding), this also counts as cancellation.

## Task

Implement a function that detects whether a single subtraction suffers from catastrophic cancellation:

```python
def spot_cancellation(a: float, b: float) -> bool:
    """
    Return True if the subtraction a - b exhibits catastrophic cancellation,
    False otherwise.
    """
```

The implementation may use the `decimal` module for high‑precision arithmetic.  
It must return a boolean value.

## Example

```python
>>> spot_cancellation(1e16 + 1, 1e16)
True          # the exact difference is 1 but float subtraction rounds to 0
>>> spot_cancellation(1000.0, 999.9999)
False         # relative error ≈ 1e-7 < 1e-12
```

## What the gate checks

The grader runs a fixed set of test cases and compares your output with a reference implementation that uses `decimal.Decimal` at high precision.  
Your solution must match the reference on all cases; otherwise the `exact_match` metric will be zero.
