## Context

Weight streaming lets you run a model whose full weight set doesn't fit in
GPU memory: the layers are stored in host (CPU) memory or on disk, and only a
sliding *window* of $K$ consecutive layers is resident on the GPU at any
instant while the runtime prefetches the next layer and evicts the oldest
one. On top of that resident window, a fixed amount of GPU memory $A$ is
always needed for activations, the KV-cache workspace, and other scratch
buffers that don't get streamed.

If layer $j$ (for $j = 0, \dots, n-1$) occupies $w_j$ bytes when resident,
then while the pipeline is at "stage" $i$ the resident window is
$\{w_i, w_{i+1}, \dots, w_{i+K-1}\}$ (clamped to $K \le n$ layers), and the
instantaneous GPU memory in use is

$$
M(i) = A + \sum_{j=i}^{i+K-1} w_j .
$$

The **minimum GPU memory required to run the whole model** is the memory
needed to survive the worst (heaviest) window the pipeline ever has to hold
resident:

$$
M_{\min} = A + \max_{0 \le i \le n-K} \sum_{j=i}^{i+K-1} w_j .
$$

## Task

Implement `min_gpu_memory`:

```python
def min_gpu_memory(layer_bytes: np.ndarray, K: int, activation_buffer: int) -> int:
    ...
```

* `layer_bytes` — 1‑D NumPy array of length $n \ge 1$, the resident size in
  bytes of each layer, in pipeline order.
* `K` — window size: how many consecutive layers are kept resident at once
  ($1 \le K$; if $K \ge n$ the whole model is resident simultaneously, i.e.
  there is only one window covering all $n$ layers).
* `activation_buffer` — fixed number of bytes always needed on top of the
  resident window (activations / scratch / KV-cache workspace).

Return the minimum total GPU memory (in bytes, as an `int`) required to run
the model, i.e. $M_{\min}$ from above: the heaviest $K$-layer sliding window
sum, plus `activation_buffer`.

## Example

```python
import numpy as np
layer_bytes = np.array([10, 40, 5, 30, 20])
K = 2
activation_buffer = 100

min_gpu_memory(layer_bytes, K, activation_buffer)
# windows of size 2: 10+40=50, 40+5=45, 5+30=35, 30+20=50
# heaviest window = 50  ->  50 + 100 = 150
# -> 150
```

## What the gate checks

A single **exact_match** gate generates several random layer-size arrays,
window sizes $K$, and activation buffers, computes $M_{\min}$ with a NumPy
sliding-window oracle (`numpy.lib.stride_tricks.sliding_window_view`), and
compares it exactly (after rounding to the nearest integer) against your
function's return value. Any mismatch or exception fails the gate.
