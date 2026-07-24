## Context

Floating‑point numbers are stored in binary as a sign bit, an exponent field and a mantissa (fraction) field.  
For a format with $e$ exponent bits and $m$ mantissa bits the bias is  

$$\text{bias} = 2^{\,e-1}-1.$$

The total number of bits is $1+e+m$.  
Typical formats used in machine learning are:

* **FP16** – 5 exponent bits, 10 mantissa bits (IEEE‑754 binary16).  
* **BF16** – 8 exponent bits, 7 mantissa bits (Brain Floating Point).  
* **E4M3** – 4 exponent bits, 3 mantissa bits.  
* **E5M2** – 5 exponent bits, 2 mantissa bits.

## Task

Implement the function `identify_fp_formats()` that returns a dictionary mapping each format name to a tuple of three integers:

```
{
    'fp16': (exp_bits, mantissa_bits, bias),
    'bf16': (exp_bits, mantissa_bits, bias),
    'E4M3': (exp_bits, mantissa_bits, bias),
    'E5M2': (exp_bits, mantissa_bits, bias)
}
```

The function must compute the bias using the formula above and not rely on any external libraries beyond the Python standard library.

## Example

```python
from solution_ref import identify_fp_formats

print(identify_fp_formats())
# {
#   'fp16':  (5, 10, 15),
#   'bf16':  (8, 7, 127),
#   'E4M3':  (4, 3, 7),
#   'E5M2':  (5, 2, 15)
# }
```

## What the gate checks

The grader verifies that the returned dictionary contains exactly the four keys above and that each tuple matches the expected values computed from the bias formula. The metric `exact_match` must equal `1.0`. No other performance or style constraints are enforced.
