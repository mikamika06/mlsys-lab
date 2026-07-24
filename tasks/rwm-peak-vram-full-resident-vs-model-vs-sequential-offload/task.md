## Context

In diffusion models, the GPU memory footprint is dominated by two components:

* **Model weights** – a tensor of shape \((n_{\text{layers}}, s)\) where \(s\) is the number of bytes per token for one layer.
* **Activations** produced during each forward pass – a buffer of size \(s \times b\), with \(b\) the batch size.

Let

$$
W = n_{\text{layers}} \cdot s,\qquad A = s \cdot b .
$$

Three common memory‑management strategies are:

1. **Full resident** – all weights and activations live on the GPU simultaneously.
2. **Model offload** – the entire model is kept in CPU RAM; only a single activation buffer resides on the GPU at any time.
3. **Sequential offload** – each layer’s weights are streamed onto the GPU one by one, so that at most one weight tensor and the activation buffer are resident.

The peak VRAM usage for each strategy can be expressed as

$$
P_{\text{full}} = \frac{W + A}{M},\qquad
P_{\text{model}} = \frac{\max(W,A)}{M},\qquad
P_{\text{seq}}   = \frac{\max(s,\,A)}{M},
$$

where \(M = 1024^2\) converts bytes to megabytes.

## Task

Implement the function `compute_peak_vram` that takes three integers:

```python
def compute_peak_vram(num_layers: int, layer_size: int, batch_size: int) -> dict:
    ...
```

* `num_layers`: number of layers in the diffusion model.
* `layer_size`: size (in bytes) of a single token for one layer.
* `batch_size`: number of tokens processed simultaneously.

The function must return a dictionary with keys

```python
{
  "full_resident":   <peak MB for full resident>,
  "model_offload":   <peak MB for model offload>,
  "sequential_offload": <peak MB for sequential offload>
}
```

All values should be `float` and expressed in megabytes. The implementation must use only pure Python arithmetic; no external libraries are required.

## Example

```python
>>> compute_peak_vram(10, 4_096, 8)
{'full_resident': 0.328125,
 'model_offload': 0.03125,
 'sequential_offload': 0.03125}
```

Here \(W = 10 \times 4096 = 40\,960\) bytes and \(A = 4\,096 \times 8 = 32\,768\) bytes, so

* \(P_{\text{full}}   = (40\,960 + 32\,768)/2^{20} = 0.328125\) MB,
* \(P_{\text{model}} = \max(40\,960, 32\,768)/2^{20} = 0.03125\) MB,
* \(P_{\text{seq}}   = \max(4\,096, 32\,768)/2^{20} = 0.03125\) MB.

## What the gate checks

The grader computes a reference peak for each strategy using the same formulas above and compares your output dictionary to this reference with exact equality (`==`). Any mismatch or exception causes the gate to fail. No other metrics are evaluated.
