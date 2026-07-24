## Context

Mixed-precision training keeps the master weights in fp32 but runs the backward
pass in fp16. IEEE binary16 has a 5-bit exponent, so its representable positive
range is only

$$6.0\times10^{-8}\;(\text{smallest subnormal}) \;\le\; |x| \;\le\; 65504\;(\text{largest finite}).$$

Real gradients routinely live around $10^{-8}$–$10^{-11}$, which is *below* the
fp16 subnormal floor. Casting them straight to fp16 flushes them to exactly zero
and the update vanishes.

**Loss scaling** fixes this. Multiply the loss by a constant $S$ before the
backward pass; by linearity every gradient is multiplied by the same $S$:

$$\frac{\partial (S\cdot L)}{\partial w} = S\cdot\frac{\partial L}{\partial w}.$$

The scaled gradients land in the middle of the fp16 range, survive the cast, and
the optimiser divides $S$ back out in fp32 before touching the weights:

$$\hat g = \frac{1}{S}\,\mathrm{fp16}\!\left(S\cdot g\right).$$

$S$ is chosen as a **power of two** so that multiplying and dividing by it only
changes the exponent field — the mantissa, and therefore the relative error, is
untouched.

## Task

Implement three functions.

```python
def pick_loss_scale(grads: np.ndarray, fp16_max: float = 65504.0) -> float: ...
def to_fp16_grads(grads: np.ndarray, scale: float) -> np.ndarray: ...
def unscale_grads(grads_fp16: np.ndarray, scale: float) -> np.ndarray: ...
```

* `pick_loss_scale` returns the **largest** $S = 2^{e}$ with integer $e$ such that

  $$\max_i |g_i| \cdot S \;\le\; \texttt{fp16\_max}.$$

  Search $e \in [-64, 64]$. Note that $e$ may be **negative**: gradients larger
  than $65504$ must be scaled *down* or they overflow to `inf`. Return `1.0` when
  `grads` is all zeros.
* `to_fp16_grads` returns `grads * scale` as a `float16` array of the same shape.
* `unscale_grads` widens a `float16` array back to `float32` and divides `scale`
  out, returning a `float32` array of the same shape.

The dtypes are part of the contract: `to_fp16_grads` must return `np.float16`
(that is the whole point — the fp16 tensor is the bottleneck), and
`unscale_grads` must return `np.float32`.

## Example

```python
import numpy as np

g = np.array([1e-8, -4e-9, 2e-10], dtype=np.float32)

# without loss scaling everything underflows
print(np.asarray(g, dtype=np.float16))
# [0. 0. 0.]

S = pick_loss_scale(g)
print(S)
# 8796093022208.0        (2**43;  1e-8 * 2**43 = 87960.9 > 65504 would overflow at 2**44)

packed = to_fp16_grads(g, S)
print(packed.dtype, packed)
# float16 [43970. -17580.   879.5]

print(unscale_grads(packed, S))
# [ 9.9977e-09 -3.9986e-09  1.9999e-10]
```

## What the gate checks

The grader builds five deterministic gradient tensors from
`np.random.default_rng(0)`, spanning magnitudes from $10^{-11}$ (deep underflow)
up to $10^{6}$ (fp16 overflow, so the chosen scale must be less than one).

* `scale_ok` — your `pick_loss_scale` must match, exactly, a power-of-two search
  the grader runs itself over $e \in [-64, 64]$ in float64. Must be `1.0`.
* `dtype_ok` — `to_fp16_grads` returns `float16` and `unscale_grads` returns
  `float32`, both with the input shape. Must be `1.0`.
* `rel_err` — the grader feeds your fp16 tensor into your `unscale_grads` and
  compares the recovered gradients with the **original fp32 gradients** using the
  global relative $L_2$ error
  $\lVert\hat g - g\rVert_2 / \lVert g\rVert_2$, taking the worst case. It must be
  at most $10^{-3}$, i.e. no worse than plain fp16 rounding noise.

The reported info metric `naive_rel_err` shows the same error for a cast to fp16
*without* loss scaling on the smallest-gradient case — it is $\approx 0.995$,
because almost every gradient was flushed to zero.
