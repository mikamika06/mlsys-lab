## Context

In many machine‑learning libraries a quantized integer type is defined by its bit width $b$ and the signed range it can represent.  
For a symmetric two’s‑complement representation the minimum and maximum values are

$$
q_{\min} = -\,2^{\,b-1}, \qquad q_{\max}= 2^{\,b-1}-1 .
$$

When packing several codes into one byte, the number of codes that fit is simply

$$
\text{pack\_factor}=\frac{8}{b},
$$

since a byte contains $8$ bits.

Typical quantized integer types used in practice are:

| dtype | bit width $b$ |
|-------|---------------|
| qint2 | 2 |
| qint4 | 4 |
| qint8 | 8 |

The task is to compute, for each of these three dtypes, the tuple $(q_{\min},\,q_{\max},\,\text{pack\_factor})$.

## Task

Implement a function `dtype_range_packing()` that returns a dictionary mapping the dtype name (e.g. `"qint2"`) to its corresponding tuple `(q_min, q_max, pack_factor)`.

```python
def dtype_range_packing() -> dict[str, tuple[int, int, int]]:
    ...
```

The function must use only pure Python arithmetic; no external libraries are required.

## Example

```python
>>> dtype_range_packing()
{
    'qint2': (-2, 1, 4),
    'qint4': (-8, 7, 2),
    'qint8': (-128, 127, 1)
}
```

## What the gate checks

The grader computes the expected dictionary from the bit widths using the formulas above and compares it to your output with an exact equality check. Any deviation – even a single integer off – will cause the gate to fail.
