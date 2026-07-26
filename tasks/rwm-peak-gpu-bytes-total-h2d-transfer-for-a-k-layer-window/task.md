## Context

In a deep‑learning pipeline each layer of a model has an associated weight tensor that must be resident on the GPU during its forward pass.  
When a model is executed, the runtime streams weights from host memory to device memory over PCIe in a *window* of size $K$: at any instant only $K$ consecutive layers are kept on the GPU.  
The **peak GPU residency** for a given window size $K$ is therefore

$$
\text{peak}(K) \;=\; \max_{i=1}^{n-K+1}\;\sum_{j=i}^{i+K-1} s_j,
$$

where $s_j$ denotes the byte size of layer $j$ and $n$ is the total number of layers.  
The **total host‑to‑device (H2D) traffic** for a forward pass is simply the sum of all layer sizes:

$$
\text{h2d}_{\text{tot}} \;=\; \sum_{j=1}^{n} s_j .
$$

Both quantities are critical for memory budgeting and bandwidth estimation on GPUs.

## Task

Implement the function `gpu_transfer_stats` that receives a list of positive integers `layer_sizes`, each representing the byte size of a layer, and an integer `window_size`.  
It must return a tuple `(peak_gpu_bytes, total_h2d_bytes)` where:

* `peak_gpu_bytes` is the maximum sum of any consecutive `window_size` layers (or the sum of all layers if `window_size > len(layer_sizes)`);
* `total_h2d_bytes` is the sum of all layer sizes.

The function must use only NumPy operations; no Python loops are allowed.  
Both returned values should be plain Python integers (`int`), not NumPy scalars.

```python
def gpu_transfer_stats(layer_sizes: list[int], window_size: int) -> tuple[int, int]:
    ...
```

## Example

```python
>>> layer_sizes = [1024, 2048, 512, 4096]
>>> window_size = 2
>>> gpu_transfer_stats(layer_sizes, window_size)
(4608, 7680)
```

Explanation:  
* Window sums are `[3072, 2560, 4608]`; the maximum is `4608`.  
* Total H2D traffic is `1024+2048+512+4096 = 7680`.

## What the gate checks

The grader computes a NumPy reference for each test case and compares it to your output.  
Your solution must return **exactly** the same tuple of integers; any mismatch causes the gate to fail.
