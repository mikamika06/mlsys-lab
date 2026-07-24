## Context

Quantizing a classifier head to `int8` is a common last-mile inference
optimization. **Per-channel** symmetric quantization gives each output
channel (each row of the weight matrix, i.e. each class) its own scale, so a
channel with small weight magnitudes isn't crushed to a handful of
quantization levels just because another channel in the same matrix has much
larger weights. For weight row $W_c \in \mathbb{R}^D$ (channel $c$ of $C$):

$$
s_c = \frac{\max_j |W_{c,j}|}{127}, \qquad
q_{c,j} = \mathrm{clip}\big(\mathrm{round}(W_{c,j} / s_c),\, -127,\, 127\big) \in \mathbb{Z},
$$

with $s_c = 1$ by convention if row $c$ is all zeros. Dequantizing recovers
an approximation $\hat{W}_{c,j} = q_{c,j} \cdot s_c$. The logits for input
$X \in \mathbb{R}^{N \times D}$ are then

$$
\hat{Y} = X \hat{W}^\top + b,
$$

with the bias $b$ kept at full precision (unquantized).

## Task

Implement `quantize_classifier_head(X, W, b)`.

- `X`: `(N, D)` float array of input activations.
- `W`: `(C, D)` float array, the classifier head's weight matrix ($C$
  classes, $D$ features).
- `b`: `(C,)` float array, the bias.

Per-channel-quantize `W` to `int8` as defined above, dequantize it, and
compute the logits. Return a 3-tuple `(logits, W_int8, scale)`:

- `logits`: `(N, C)` float array, $X \hat{W}^\top + b$.
- `W_int8`: `(C, D)` integer array (values in $[-127, 127]$), the quantized
  weights $q$.
- `scale`: `(C,)` float array, the per-channel scales $s_c$.

## Example

```python
import numpy as np

X = np.array([[1.0, 2.0]])
W = np.array([[4.0, -2.0], [0.5, 0.5]])
b = np.array([0.0, 0.0])

logits, W_int8, scale = quantize_classifier_head(X, W, b)
# scale ~= [4/127, 0.5/127] -- each row gets its OWN scale
```

## What the gate checks

The gate builds a classifier head where different output channels have
deliberately very different weight magnitudes (so per-tensor and per-channel
quantization diverge), runs a batch of inputs through it, and checks two
things independently:

1. **`argmax_agreement`**: the fraction of rows where your `logits`' top-1
   class matches the top-1 class of the un-quantized reference `X @ W.T +
   b`, computed directly with NumPy. Must be $\ge 0.98$.
2. **`quant_valid`**: `1.0` only if *all* of the following hold, else `0.0`:
   - `W_int8` holds integers in $[-127, 127]$ with the right shape;
   - `scale` matches the independently-computed per-channel oracle scale
     $s_c = \max_j|W_{c,j}| / 127$ to within $10^{-6}$ relative error (this
     is what rules out a single global (per-tensor) scale, or any other
     shortcut, sneaking past the `argmax_agreement` check);
   - `logits` is actually reconstructed from the *returned* `W_int8` and
     `scale` (i.e. `logits ≈ X @ (W_int8 * scale[:, None]).T + b`), so
     returning the un-quantized reference logits directly cannot pass.

Both gates must hold.
