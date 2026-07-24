## Context

In many deep‑learning frameworks a model is split across several devices. A common strategy is to place each layer on the first device that has enough free memory, filling GPUs first, then CPU, and finally falling back to disk if no device can accommodate the layer. This greedy sequential allocation respects the order of layers and the capacity limits of every device.

Let $L = (l_1,\dots,l_n)$ be the list of memory footprints of the model’s layers in megabytes. Let $G=(g_0,\dots,g_{m-1})$ be the capacities of $m$ GPUs, and let $C$ be the capacity of the CPU. The algorithm proceeds as follows:

For each layer size $l_i$:
  * Find the smallest index $j$ such that $g_j \ge l_i$; if found, assign the layer to GPU $j$ and reduce $g_j$ by $l_i$.
  * Otherwise, if $C \ge l_i$, assign it to the CPU and reduce $C$ by $l_i$.
  * If neither device can fit the layer, place it on disk (unlimited capacity).

This is a deterministic algorithm that yields a unique assignment for any input.

## Task

Implement the function `assign_layers`:

```python
def assign_layers(layer_sizes: list[int], gpu_caps: list[int], cpu_cap: int) -> list[str]:
    ...
```

The function receives:
- `layer_sizes`: memory usage of each layer in megabytes.
- `gpu_caps`: capacities of each GPU, in the same order as they should be considered.
- `cpu_cap`: capacity of the CPU.

It must return a list of strings describing the device chosen for every layer. Use `"gpu{i}"` (where *i* is the zero‑based index) for GPUs, `"cpu"` for the CPU, and `"disk"` when no device can accommodate the layer.

The implementation should be pure Python; no external libraries are required.

## Example

```python
layer_sizes = [200, 400, 150, 800]
gpu_caps   = [500, 300]     # GPU0 has 500 MB, GPU1 has 300 MB
cpu_cap    = 600           # CPU can hold 600 MB

assign_layers(layer_sizes, gpu_caps, cpu_cap)
# ['gpu0', 'gpu0', 'gpu1', 'disk']
```

Explanation:  
* Layer 1 (200 MB) fits on GPU 0 → remaining GPU 0 capacity 300 MB.  
* Layer 2 (400 MB) also fits on GPU 0 → remaining 300 MB – now insufficient for the next layer.  
* Layer 3 (150 MB) fits on GPU 1 → remaining 150 MB.  
* Layer 4 (800 MB) does not fit on any device, so it is placed on disk.

## What the gate checks

The grader computes a reference assignment using the same greedy algorithm described above and compares it to your output. The metric `exact_match` must equal `1.0`. Any deviation causes the task to fail.
