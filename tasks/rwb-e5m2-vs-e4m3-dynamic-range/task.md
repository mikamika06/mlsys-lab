## Context

Floating‑point numbers are represented by a sign bit, an exponent field of $e$ bits and a mantissa (fraction) field of $m$ bits.  
The bias for the exponent is  

$$b = 2^{\,e-1} - 1.$$

For a normalised number the value is

$$x = \pm\bigl(1 + f\bigr)\,2^{\,E-b},$$

where $f$ is the fraction represented by the mantissa bits and $E$ is the unsigned integer stored in the exponent field.  
The largest normal value occurs when all mantissa bits are 1 (so $f = 1-2^{-m}$) and the exponent field is the maximum normal value, i.e. all ones except the reserved pattern for infinities/NaNs:

$$E_{\max}=2^{\,e}-2.$$

Hence

$$\text{max}_{\text{normal}}=(2-2^{-m})\,2^{\,E_{\max}-b}.$$

The smallest normal value uses an exponent field of 1 (the lowest non‑subnormal exponent) and a zero mantissa:

$$\text{min}_{\text{normal}}=1 \cdot 2^{\,1-b}.$$

Two concrete formats are considered in this task:

* **E4M3** – $e=4$, $m=3$, bias $b=7$  
* **E5M2** – $e=5$, $m=2$, bias $b=15$

The exact values for these two formats can be derived from the formulas above.

## Task

Implement a function with the following signature:

```python
def dynamic_range() -> tuple[float, float, float, float]:
    ...
```

It must return a 4‑tuple of `float64` numbers in the order  
`(max_e4m3, min_e4m3, max_e5m2, min_e5m2)`.  
The implementation should use only arithmetic operations; no loops or external libraries are required beyond standard Python.

## Example

```python
>>> from your_module import dynamic_range
>>> dynamic_range()
(240.0, 0.015625, 57344.0, 6.103515625e-05)
```

The output matches the exact values derived in the context section.

## What the gate checks

* **Relative error** – The returned tuple is compared to a reference computed by an oracle that implements the formulas above.  
  The metric `rel_err` must satisfy  

  $$\mathrm{rel\_err} \le 10^{-9}.$$

  This ensures numerical correctness to within one part in a billion.

* **No side effects** – The function should not modify global state or rely on external randomness.

The gate is satisfied only when the relative error of all four numbers is below the threshold.
