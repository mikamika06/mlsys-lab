## Context

In fully sharded data‑parallel training each rank holds a fraction of the model parameters, gradients and optimizer state. The peak memory on a GPU is therefore the sum of the sharded tensors plus any transient buffers that are materialised during forward or backward passes. A common transient buffer is an all‑gather of the activations (or gradients) for the largest layer; its size is independent of the number of ranks.

Let

- $L_i$ be the number of elements in layer $i$,  
- $\ell_{\max}=\max_i L_i$,
- $s$ the element size in bytes ($\mathrm{float32}\to4,\;\mathrm{float64}\to8$, etc.),
- $R$ the number of ranks.

The total number of parameter elements is $P=\sum_i L_i$.  
With an optimizer such as Adam each rank also stores two auxiliary tensors per parameter, so the sharded part occupies

$$\frac{3\,P}{R} \; s.$$

If CPU off‑load is enabled all these tensors are moved to host memory and only the transient buffer remains on the device:

$$M_{\text{off}} = \ell_{\max}\,s.$$

Otherwise the peak device memory is

$$M_{\text{on}} = \frac{3\,P}{R} s + \ell_{\max}\,s.$$

## Task

Implement `peak_memory_per_rank` that takes:

```python
def peak_memory_per_rank(layer_sizes: list[int],
                         dtype: str = "float32",
                         cpu_offload: bool = False,
                         num_ranks: int = 1) -> int:
    ...
```

- `layer_sizes` – the number of elements in each layer.  
- `dtype` – NumPy dtype string (`"float16"`, `"float32"`, …).  
- `cpu_offload` – whether sharded states are off‑loaded to CPU.  
- `num_ranks` – total number of ranks.

The function must return the peak memory in **bytes** as an integer, computed exactly according to the formulas above.

## Example

```python
import numpy as np
# two layers: 1000 and 1500 elements, float32, no off‑load, 2 ranks
peak = peak_memory_per_rank([1000, 1500], dtype="float32",
                            cpu_offload=False, num_ranks=2)
print(peak)   # 3 * (2500*4)/2 + 1500*4 = 12000 bytes
```

## What the gate checks

The grader recomputes the reference value using NumPy and compares it to your output with an exact integer match. No other metrics are used.
