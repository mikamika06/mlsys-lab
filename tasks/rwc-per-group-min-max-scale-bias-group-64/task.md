## Context

In many quantisation schemes a weight tensor is split into *groups* of fixed size and each group is represented by a small number of bits.  
For a group \(g\) with elements \(\{w_i\}\) we compute the minimum and maximum values

$$m_g = \min_{i} w_i,\qquad M_g = \max_{i} w_i.$$

If we want to encode each element using \(b\) bits, the quantisation step (scale) is

$$s_g = \frac{M_g - m_g}{2^b-1},$$

and a bias term that shifts the zero‑point back into the original range is simply

$$c_g = m_g.$$

The quantised value of an element \(w_i\) in group \(g\) is then

$$\hat{w}_i = \operatorname{round}\!\left(\frac{w_i - c_g}{s_g}\right).$$

This task asks you to implement the forward pass that produces the per‑group scale and bias vectors for a 1‑D weight array.

## Task

Implement `compute_group_params(weights, group_size=64, bits=4)`:

```python
def compute_group_params(weights: np.ndarray,
                         group_size: int = 64,
                         bits: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

* `weights` is a one‑dimensional NumPy array of arbitrary length that is a multiple of `group_size`.
* The function must return two 1‑D arrays:
  * `scales`: the per‑group scale \(s_g\) as defined above.
  * `biases`: the per‑group bias \(c_g = m_g\).
* Both output arrays should be of type `float64` and have length equal to the number of groups.

The implementation must use NumPy vectorised operations only; no explicit Python loops are allowed.

## Example

```python
import numpy as np

weights = np.array([0.1, 0.2, -0.5, 1.3] * 16)   # length 64
scales, biases = compute_group_params(weights)
print(scales.shape, biases.shape)               # (1, 1)
print(scales[0], biases[0])                    # scale and bias for the single group
```

## What the gate checks

The grader computes a reference implementation using NumPy.  
It then compares your `scales` and `biases` against the reference with the metric

$$\max_{\text{group}}\bigl(\,|s_g^{\text{ref}}-s_g^{\text{sol}}|,\; |c_g^{\text{ref}}-c_g^{\text{sol}}|\bigr).$$

The solution must satisfy `max_abs_err <= 1e‑12`.  
Any deviation (including wrong shapes, dtypes or integer division) will cause the gate to fail.
