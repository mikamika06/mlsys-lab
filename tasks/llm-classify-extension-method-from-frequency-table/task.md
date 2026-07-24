## Context

The Rotary Position Embedding (RoPE) used in transformer models encodes positional information by multiplying each token embedding with a complex exponential whose frequency depends on the dimension index $i$ and the model dimensionality $d$.  
For a standard RoPE, the inverse frequencies are defined as

$$
\text{inv\_freq}_i = \frac{1}{10000^{\,2i/d}}, \qquad i=0,\dots,d-1 .
$$

Several extensions modify this table:

* **PI (Position‑Interpolation)** – all frequencies are multiplied by a constant factor $k\neq 1$.
* **NTK** – the exponent is reduced from $2$ to $1$, giving  
  $$ \text{inv\_freq}_i = \frac{1}{10000^{\,i/d}} .$$
* **YaRN (Yarn‑like)** – frequencies are linearly interpolated between the first and last values, i.e.  
  $$ \text{inv\_freq}_i = \text{inv\_freq}_0 + i\;\frac{\text{inv\_freq}_{d-1}-\text{inv\_freq}_0}{d-1} .$$
* **None** – the standard RoPE frequencies.

The task is to infer which extension was used given only the array of inverse frequencies.

## Task

Implement `classify_extension(inv_freq)`:

```python
def classify_extension(inv_freq: np.ndarray) -> str:
    ...
```

It receives a 1‑D NumPy array of shape `(n,)` containing the inverse frequency table and returns one of the strings `"PI"`, `"NTK"`, `"YaRN"` or `"None"`.

The implementation must:

* work for any positive integer `n`.
* be tolerant to small floating‑point errors (relative tolerance $10^{-6}$).
* not use explicit Python loops over the array elements.
* run in constant time relative to `n` (i.e. linear operations only).

## Example

```python
import numpy as np

def standard_inv_freq(n):
    return 1 / (10000 ** (2 * np.arange(n) / n))

def pi_inv_freq(n, k=2.0):
    return k * standard_inv_freq(n)

def ntk_inv_freq(n):
    return 1 / (10000 ** (np.arange(n) / n))

def yarn_inv_freq(n):
    std = standard_inv_freq(n)
    return np.linspace(std[0], std[-1], n)

n = 5
print(classify_extension(standard_inv_freq(n)))   # "None"
print(classify_extension(pi_inv_freq(n, k=3.0)))   # "PI"
print(classify_extension(ntk_inv_freq(n)))        # "NTK"
print(classify_extension(yarn_inv_freq(n)))       # "YaRN"
```

## What the gate checks

The grader computes a reference classification using the same logic as the reference solution and compares it to the candidate's output.  
A single metric `exact_match` is used: the candidate passes if its returned string equals the reference for all test cases.
