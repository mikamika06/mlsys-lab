## Context

In many quantized neural‑network libraries a weight tensor is stored as signed 4‑bit integers (qint4). Two such values are packed into one byte: the lower nibble holds the first value and the upper nibble the second. The mapping from a signed integer $v\in[-8,7]$ to an unsigned 4‑bit code $c$ is

$$
c = \begin{cases}
v & v\ge 0\\[2pt]
v+16 & v<0
\end{cases},
$$

and the dequantisation back to a floating point number uses a per‑axis scale factor $\alpha>0$

$$
x = \alpha\, v .
$$

The task is to recover the original floating point values from a packed byte array and a scalar scale.

## Task

Implement `unpack_dequant_qint4(packed: np.ndarray, scale: float) -> np.ndarray`:

```python
def unpack_dequant_qint4(packed: np.ndarray, scale: float) -> np.ndarray:
    ...
```

* `packed` is a 1‑D NumPy array of type `np.uint8`. Each element contains two qint4 codes as described above.
* The function must return a 1‑D NumPy array of type `np.float32` containing the dequantised values in the order they appear in the packed buffer (low nibble first, then high nibble).
* No Python loops are allowed; use vectorised NumPy operations only.

## Example

```python
import numpy as np
# two qint4 values: 3 and 5
packed = np.array([0x53], dtype=np.uint8)   # low=0x3 (3), high=0x5 (5)
scale = 0.1
out = unpack_dequant_qint4(packed, scale)
print(out)          # [0.3 0.5]
```

## What the gate checks

The grader computes a reference dequantisation using NumPy and compares it to your output with the scorer `max_abs_err`. The maximum absolute difference must be at most $10^{-6}$.
