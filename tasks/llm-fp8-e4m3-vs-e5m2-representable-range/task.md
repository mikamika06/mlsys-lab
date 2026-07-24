## Context

Binary floating‑point formats are defined by a sign bit, an exponent field of $e$ bits and a fraction (mantissa) field of $m$ bits.  
For IEEE‑style binary formats the bias is  

$$\text{bias} = 2^{\,e-1}-1,$$

the largest normal exponent field value is $2^e-2$ (all ones are reserved for Inf/NaN), and the smallest normal exponent field value is $1$.  
With these conventions the numerical range of a format can be expressed exactly:

* **Largest finite value**  

$$
V_{\max} = \bigl( 2 - 2^{-m}\bigr)\; 2^{\,E_{\max}-\text{bias}},
$$

where $E_{\max}=2^e-2$.

* **Smallest normal positive value**  

$$
V_{\min}^{\text{norm}} = 2^{\,1-\text{bias}} .
$$

* **Smallest subnormal positive value** (if the format has subnormals)  

$$
V_{\min}^{\text{sub}} = 2^{\,1-\text{bias}-m}.
$$

The representable set is therefore  

$$
\{-x,0,x: x \in [V_{\min}^{\text{sub}}, V_{\max}]\}\cup\{\pm V_{\min}^{\text{norm}}\},
$$

with the convention that $\pm\infty$ and NaN are *not* representable.

Two popular 8‑bit formats used in modern language models are:

| Format | $e$ | $m$ | Bias | $V_{\max}$ | $V_{\min}^{\text{sub}}$ |
|--------|-----|-----|------|------------|--------------------------|
| e4m3   | 4   | 3   | 7    | $240$      | $2^{-9}\approx1.95\times10^{-3}$ |
| e5m2   | 5   | 2   | 15   | $57344$    | $2^{-14-2}=1.53\times10^{-5}$ |

Thus the e5m2 format can represent a much wider range of magnitudes than e4m3.

## Task

Implement the function

```python
def fp8_representability(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

* `values` is any NumPy array of real numbers (float32 or float64).  
* The function must return a tuple `(mask_e4m3, mask_e5m2)` where each element is a boolean array of the same shape as `values`.  
  * `mask_e4m3[i]` is `True` iff `values[i]` can be represented exactly in the e4m3 format.  
  * `mask_e5m2[i]` is `True` iff `values[i]` can be represented exactly in the e5m2 format.  

The implementation must use only NumPy vectorised operations; no explicit Python loops are allowed.

## Example

```python
import numpy as np
from fp8 import fp8_representability  # your solution will live here

A = np.array([0, 1e-4, 0.02, 100, 200, 250, 300, np.nan, np.inf])
mask_e4m3, mask_e5m2 = fp8_representability(A)

print(mask_e4m3)
# [ True  True False  True  True False False False False]

print(mask_e5m2)
# [ True  True  True  True  True  True  True False False]
```

In this example, `250` is too large for e4m3 but fits in e5m2; subnormal values such as `1e-4` are representable only in e5m2.

## What the gate checks

The grader computes a reference solution using the exact formulas above and compares your output element‑wise.  
If every boolean value matches, the metric `exact_match` is 1.0; otherwise it is 0.0.  
No other performance or style metrics are enforced for this task.
