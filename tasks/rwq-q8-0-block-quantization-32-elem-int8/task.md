## Context

Block quantization is a common technique in deep learning to reduce the memory footprint of weight tensors.  
The Q8_0 format used by many inference engines stores each element as an **int8** code and reconstructs it with a per‑block scale factor.

For a block of $32$ consecutive elements $x_i$, let

$$
a_{\max} = \max_{i}\lvert x_i\rvert .
$$

If $a_{\max}=0$ we set the scale $d=1$.  
Otherwise

$$
d = \frac{a_{\max}}{127}.
$$

The quantized code is

$$
q_i = \operatorname{round}\!\left(\frac{x_i}{d}\right),
$$

clipped to the signed 8‑bit range $[-127,\,127]$.  
Dequantization simply multiplies back:

$$
\hat{x}_i = q_i \cdot d .
$$

The block size is fixed at $32$ elements; each block gets its own scale.

## Task

Implement a function that performs this Q8_0 quantization on an arbitrary 1‑D NumPy array.

```python
import numpy as np
from typing import Tuple

def q8_0_quantize(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize `arr` using the Q8_0 format (32‑element blocks, int8 codes).

    Parameters
    ----------
    arr : np.ndarray
        1‑D array of arbitrary length and dtype convertible to float64.

    Returns
    -------
    codes : np.ndarray
        int8 array of the same shape as `arr` containing the quantized codes.
    dequant : np.ndarray
        float64 array of the same shape as `arr` containing the reconstructed values.
    """
    ...
```

The function must:

1. Work for any 1‑D array length (including lengths not divisible by $32$).
2. Return a NumPy array of dtype `np.int8` for the codes and a `float64` array for the dequantized values.
3. Use only NumPy operations; no explicit Python loops over individual elements.

## Example

```python
import numpy as np

arr = np.array([0.0, 1.5, -2.7, 0.9], dtype=np.float32)
codes, deq = q8_0_quantize(arr)

print(codes)   # e.g., [  0  64 -96  48]
print(deq)     # reconstructed values close to the original
```

The exact codes depend on the block scale; for a single‑block array the scale is computed from all four elements.

## What the gate checks

Two metrics are evaluated:

* **Relative error** – `rel_err` between the dequantized output and the original input, defined as  
  $$\mathrm{rel\_err} = \frac{\lVert \hat{x}-x\rVert}{\lVert x\rVert + 10^{-12}}.$$
  The gate requires $\mathrm{rel\_err}\leq 2\times10^{-2}$.

* **Code correctness** – the returned codes must be `int8` and match the exact Q8_0 algorithm.  
  Any mismatch results in a large error that fails the relative‑error check.

The grader generates several random arrays, runs your implementation, compares it against an oracle, and reports the worst relative error found. A correct solution will achieve an error well below $2\times10^{-2}$.
