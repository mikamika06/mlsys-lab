## Context

A naive ("eager mode") implementation of an activation epilogue runs each
elementwise op as its own kernel launch, writing a full-size intermediate
array back to memory after every stage and reading it back in for the
next. Real inference kernels instead perform **vertical (pointwise)
fusion**: they compose the whole chain of elementwise ops into a single
per-element formula and evaluate it in one pass, reading each input once
and writing the output once. The MATH is identical either way — fusion is
purely about how many times data crosses the memory boundary — so the
fused and unfused versions must produce numerically identical results.

A common transformer MLP epilogue chains:

$$
h = X + \text{bias} \qquad
h = \operatorname{gelu}(h) \qquad
h = h + \text{residual} \qquad
h = h \cdot \text{scale}
$$

using the tanh-approximation GELU (as used by GPT-2/BERT):

$$
\operatorname{gelu}(x) = \tfrac{1}{2}x\left(1 + \tanh\!\left(\sqrt{2/\pi}\,\left(x + 0.044715\,x^3\right)\right)\right) .
$$

## Task

Implement `fused_elementwise_chain`:

```python
def fused_elementwise_chain(X: np.ndarray, bias: np.ndarray, residual: np.ndarray, scale: float) -> np.ndarray:
    ...
```

* `X` — `(batch, dim)` activations.
* `bias` — `(dim,)`, broadcasts over the batch axis.
* `residual` — `(batch, dim)`.
* `scale` — python `float`.

Compute the chain above — bias-add, tanh-GELU, residual-add, scale — as a
**single fused expression**: build the whole per-element formula and
evaluate it in one shot, rather than writing each stage's full result out
to a separate named intermediate array and reading it back for the next
stage. Return an array the same shape as `X`.

## Example

```python
import numpy as np

X = np.zeros((1, 3))
bias = np.array([0.0, 0.0, 0.0])
residual = np.array([[1.0, 2.0, 3.0]])
scale = 2.0

fused_elementwise_chain(X, bias, residual, scale)
# gelu(0) == 0, so this reduces to residual * scale -> [[2.0, 4.0, 6.0]]
```

## What the gate checks

A single gate, **max_abs_err**, compares your output against an
op-by-op reference (each stage computed and materialized separately, in
`float64`) across 6 random `(X, bias, residual, scale)` combinations of
varying batch size and feature dimension. Must be `<= 1e-6`.
