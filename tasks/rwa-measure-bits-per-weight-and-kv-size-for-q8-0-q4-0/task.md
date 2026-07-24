## Context

Quantization formats such as Q8_0 and Q4_0 store a block of $32$ weights in a compact byte representation.  
For Q8_0 the block occupies $34$ bytes, for Q4_0 it occupies $18$ bytes.  The *bits per weight* (bpw) is therefore

$$
\text{bpw} = \frac{\text{block\_bytes}\times 8}{32}.
$$

The key–value (KV) size of a block is the number of bytes required to store it relative to a full‑precision FP16 value ($2$ bytes per weight).  The ratio is

$$
\text{kv\_ratio} = \frac{\text{block\_bytes}}{32\times 2}.
$$

These two metrics are useful for estimating memory usage and compression efficiency in transformer models.

## Task

Implement the function `measure_qkv()` that returns a tuple of four floating‑point numbers:

```python
def measure_qkv() -> tuple[float, float, float, float]:
    """
    Returns:
        bpw_q8   – bits per weight for Q8_0
        bpw_q4   – bits per weight for Q4_0
        kv_ratio_q8 – KV size ratio for Q8_0 relative to FP16
        kv_ratio_q4 – KV size ratio for Q4_0 relative to FP16
    """
```

The function should perform the calculations exactly as described in the context section, using floating‑point division.  No external libraries are required.

## Example

```python
>>> from your_module import measure_qkv
>>> measure_qkv()
(8.5, 4.5, 0.53125, 0.28125)
```

The output shows that Q8_0 uses $8.5$ bits per weight and occupies about $53\%$ of the space required by FP16, while Q4_0 uses $4.5$ bits per weight and occupies about $28\%$.

## What the gate checks

A single numerical gate compares the returned tuple to a reference computed by an oracle.  
The relative L2 error between the candidate array and the reference must satisfy

$$
\frac{\lVert \text{candidate} - \text{reference}\rVert_2}
     {\lVert \text{reference}\rVert_2 + 10^{-12}}
   \le 10^{-9}.
$$

If this condition holds, the solution passes; otherwise it fails.
