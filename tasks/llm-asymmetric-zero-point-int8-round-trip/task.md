## Context

Quantization is a common technique in deep learning to reduce model size and inference latency by representing weights and activations with lower‑precision integers.  
For an asymmetric scheme we map real values $x$ into the integer range $\mathcal{I}=\{-128,\dots,127\}$ using a scale factor $s>0$ and a zero point $z\in \mathbb{Z}\cap[-128,127]$:

$$
q = \operatorname{clip}\!\bigl(\operatorname{round}\!\bigl(\tfrac{x}{s}+z\bigr),-128,127\bigr),
$$

where $\operatorname{clip}$ enforces the bounds.  
The dequantized value is recovered by

$$
\hat x = (q - z)\,s .
$$

Choosing $s$ and $z$ so that the quantization range covers the data’s minimum and maximum values yields minimal distortion.  For a real array $x$, let $\min(x)$ and $\max(x)$ denote its smallest and largest entries.  The optimal parameters are

$$
s = \frac{\max(x)-\min(x)}{255}, \qquad
z = \operatorname{round}\!\bigl(-\,\tfrac{\min(x)}{s}\bigr).
$$

The goal of this task is to implement a round‑trip that takes a list, quantizes it with the above asymmetric scheme, then immediately dequantizes it back to floating point.  The function must return both the dequantized array and the integer zero point used.

## Task

Implement `asymmetric_quant_round_trip(x)`:

```python
def asymmetric_quant_round_trip(x: list[float]) -> tuple[list[float], int]:
    ...
```

* `x` is a list of arbitrary shape containing floating‑point values.  
* The function must use only vectorized Python operations; no explicit Python loops are allowed.  
* It should return a tuple `(dequantized, zero_point)` where:
  * `dequantized` has the same shape as `x`, dtype `float64`, and contains the dequantized values.
  * `zero_point` is an integer in the range $[-128,127]$ that was used during quantization.

## Example

```python
from your_module import asymmetric_quant_round_trip

A = [0.0, 1.0, -2.5]
deq, zp = asymmetric_quant_round_trip(A)
print(deq)   # approximately [0., 1., -2.5]
print(zp)    # integer zero point used
```

## What the gate checks

Two metrics are evaluated:

* **rel_err** – The global relative L2 error between the original array and the dequantized result:
  $$
  \mathrm{rel\_err} = \frac{\lVert \hat x - x\rVert_2}{\lVert x\rVert_2 + 10^{-12}} .
  $$
  The solution must achieve $\mathrm{rel\_err}\le 1\times10^{-2}$ on all test cases.

* **zero_point_match** – A binary check that the returned zero point equals the exact integer computed by the oracle.  It must be exactly equal for every test case; otherwise the metric is $0$.

Both metrics are required to pass for the solution to be accepted.
