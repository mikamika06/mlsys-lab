## Context

Average Memory Access Time (AMAT) measures the expected latency of a memory access in a cache hierarchy.  
For a multi‑level cache with miss penalties $t_{\text{L2}}, t_{\text{L3}},\dots$ and hit probabilities $h_i$, the classic formula for three levels is  

$$
\mathrm{AMAT}= t_{\text{L1}}
   + (1-h_1)\!\left(t_{\text{L2}}+(1-h_2)\!\bigl(t_{\text{L3}}+(1-h_3)t_{\text{DRAM}}\bigr)\right).
$$  

In this task the latencies are fixed:  

$ t_{\text{L1}}=1,\quad
  t_{\text{L2}}=4,\quad
  t_{\text{L3}}=12,\quad
  t_{\text{DRAM}}=100.$

These numbers correspond to cycle counts and are identical for every test case.

## Task

Implement the function `compute_amat`:

```python
def compute_amat(hit_rates: np.ndarray) -> float:
    ...
```

* `hit_rates` is a one‑dimensional sequence of length 3 containing hit probabilities for L1, L2 and L3 (values in $[0,\,1]$).  
* The function should return the AMAT in cycle units as a scalar floating point number.  
* No loops over cache levels are required; you may express the result with straight arithmetic.

## Example

```python
import numpy as np
hr = np.array([0.98, 0.95, 0.90])
amat = compute_amat(hr)
print(amat)   # 1.102
```

(The calculation is  
$1 + 0.02\,(4 + 0.05\,(12 + 0.10\times100)) \;=\; 1.102.$)

## What the gate checks

* **Metric**: `rel_err` – the relative error between your result and a reference implementation computed with the same fixed latencies.  
  The maximum relative error over all provided hit‑rate rows must be $\le 10^{-9}$.

No other constraints are enforced; correctness is measured solely by numerical accuracy.
