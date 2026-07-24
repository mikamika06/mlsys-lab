## Context

FP8 KV-cache quantization stores `K`/`V` as scaled 8-bit floats (e4m3: 4
exponent bits, 3 mantissa bits, max finite magnitude $448$). For a tensor
$X$, scaled quantization picks

$$
s = \frac{\max(|X|)}{448}, \qquad \hat{X} = s \cdot \operatorname{fp8e4m3}\!\left(\frac{X}{s}\right).
$$

The scale $s$ can be computed **per-tensor** (one scale shared by every
attention head) or **per-head** (a separate $s_h$ for each head $h$,
using only that head's own values). When one head has a much larger value
range than the rest — an *outlier head*, common in real KV caches — a
per-tensor scale is dragged up by that head's `amax`, leaving far fewer
of e4m3's 8 representable "steps" available to every other, quieter head.
A per-head scale gives every head the full dynamic range of e4m3
regardless of what any other head is doing.

Attention is

$$
\operatorname{Attn}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V,
$$

computed independently per head. The end-to-end effect of a scaling
strategy is measured by comparing its attention output against attention
run on the original, unquantized `K`/`V`, **per head**, then averaging
equally across heads:

$$
\Delta = \frac{1}{H}\sum_{h=1}^{H} \frac{\lVert O_h^{\text{quant}} - O_h^{\text{fp32}} \rVert_2}{\lVert O_h^{\text{fp32}} \rVert_2} .
$$

Averaging per-head (rather than one global norm over the whole tensor)
matters here: a single dominant outlier head's huge magnitude would
otherwise swamp a global norm and hide exactly the degradation this task
is measuring in the quieter heads.

## Task

Implement `kv_scale_granularity_delta(Q, K, V)`:

```python
def kv_scale_granularity_delta(Q, K, V):
    ...
```

Inputs are NumPy arrays of shape $(H, M, D)$ for `Q` and $(H, N, D)$ for
`K`/`V`.

1. Compute the exact reference output $O^{\text{fp32}}$ by running
   attention on the original `Q`, `K`, `V` in `float64`.
2. Quantize-and-dequantize `K` and `V` to simulated e4m3 with a single
   **per-tensor** scale ($s = \max(|X|)/448$ over the whole tensor), run
   attention with the reconstructed `K`/`V`, and compute $\Delta$
   (per-head-averaged relative error, as above) against $O^{\text{fp32}}$.
3. Do the same with a **per-head** scale ($s_h = \max(|X_h|)/448$,
   computed independently for each head).
4. Return `(per_tensor_delta, per_head_delta)` as Python floats.

## Example

```python
import numpy as np

Q = np.load("fixtures/q.npy")
K = np.load("fixtures/k.npy")
V = np.load("fixtures/v.npy")

per_tensor_delta, per_head_delta = kv_scale_granularity_delta(Q, K, V)
# per_head_delta < per_tensor_delta -- the fixture has one outlier-scale
# head, so a shared per-tensor scale visibly hurts the other heads more
# than giving every head its own scale does.
```

## What the gate checks

The gate loads the committed `q.npy`/`k.npy`/`v.npy` fixture (one head
scaled far above the rest) and computes its own per-tensor and per-head
deltas independently, using the same e4m3 simulation and per-head-averaged
error definition. It fails (reported as an infinite error) if
`per_head_delta` is not strictly smaller than `per_tensor_delta`, or if
either of your reported values doesn't closely match the oracle's own
computed value. Using a single global-norm error (instead of averaging
per-head) or always reporting one scale as universally better regardless
of the data will not match the oracle's numbers.
