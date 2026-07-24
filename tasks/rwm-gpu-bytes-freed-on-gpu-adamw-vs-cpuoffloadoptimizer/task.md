## Context

In deep learning training, the optimizer state (momentum buffers, running averages, etc.) and gradients are typically stored on the GPU to avoid costly data transfers during back‑propagation. A CPU‑offloaded optimizer moves these tensors off the GPU into host memory, freeing VRAM for larger models or batch sizes.

Let

- $K$ be the number of trainable parameters,
- $\phi$ the size in bytes of a single parameter tensor (e.g. 4 for `float32`, 2 for `float16`),
- $\texttt{offload\_gradients}$ a boolean flag indicating whether gradients are also moved to CPU.

The amount of GPU memory that becomes available when switching from an on‑GPU optimizer such as AdamW to a CPU‑offloaded variant is

$$
\text{freed} = K \cdot \phi + 2K \cdot \phi \;\mathbf{1}_{\{\texttt{offload\_gradients}\}}
= K \cdot \phi \bigl(1 + 2\,\mathbf{1}_{\{\texttt{offload\_gradients}\}}\bigr).
$$

The first term accounts for the optimizer state; the second term (if enabled) adds the two gradient tensors per parameter.

## Task

Implement the function `gpu_bytes_freed` that computes the number of bytes freed on the GPU when switching from AdamW to a CPU‑offloaded optimizer:

```python
def gpu_bytes_freed(K: int, phi: int, offload_gradients: bool) -> int:
    ...
```

The function must return an integer number of bytes.

## Example

```python
>>> gpu_bytes_freed(1000, 4, False)
4000
>>> gpu_bytes_freed(1000, 4, True)
12000   # 1000*4 + 2*1000*4 = 4000 + 8000
```

## What the gate checks

The grader evaluates your implementation against a NumPy‑based oracle that generates random test cases. The metric `exact_match` requires that your function returns exactly the same integer for every case.
