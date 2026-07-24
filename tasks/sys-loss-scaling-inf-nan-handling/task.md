## Context

When training in `float16`, small gradients can underflow to zero. Mixed
precision training avoids this with **loss scaling**: multiply the loss by a
large factor $s$ (a power of two, e.g. $s = 2^{16}$) before the backward
pass, so gradients are scaled up by $s$ too and stay in `float16`'s
representable range. Before the optimizer step, the gradients must be
**unscaled** by dividing by $s$:

$$
g = \frac{g_{\text{scaled}}}{s} .
$$

Scaling up can also push some gradients past `float16`'s max finite value,
producing `inf`, or an `inf` can combine with another value to produce `nan`.
When this happens, the whole optimizer step for that iteration must be
**skipped** — applying a corrupted update would poison the model — and the
scale factor $s$ would typically be reduced for the next iteration (a
dynamic loss scaler). This task only covers the per-step decision, not the
scale-adjustment schedule.

Because $s$ is always a power of two, dividing a finite `float32` gradient by
$s$ is an exact operation (it only shifts the exponent) — it introduces no
rounding error, so a correct implementation's unscaled gradients are
bit-for-bit identical to the reference, not merely close.

## Task

Implement `unscale_and_check(scaled_grads, scale)`.

- `scaled_grads`: a list of `float32` NumPy arrays (the scaled gradients for
  one iteration, one array per parameter tensor). Any array may contain
  `inf` or `nan` entries, indicating overflow happened during the scaled
  backward pass.
- `scale`: a Python `float`, the loss-scale factor $s$ (always a power of
  two in the tests).

Return a tuple `(skip, unscaled_grads)`:

- `skip`: a Python `bool`, `True` if **any** element in **any** array of
  `scaled_grads` is `inf` or `nan`, `False` otherwise.
- `unscaled_grads`: a list of `float32` NumPy arrays, same shapes as
  `scaled_grads`, where `unscaled_grads[i] = scaled_grads[i] / scale`
  elementwise — computed unconditionally, whether or not `skip` is `True`
  (this mirrors real automatic-mixed-precision scalers, which always divide
  and let the caller decide separately whether to skip the optimizer step).

## Example

```python
import numpy as np

grads = [np.array([2.0, 4.0], dtype=np.float32), np.array([np.inf], dtype=np.float32)]
skip, unscaled = unscale_and_check(grads, 2.0)
# skip == True
# unscaled == [array([1.0, 2.0], dtype=float32), array([inf], dtype=float32)]
```

## What the gate checks

The gate runs `unscale_and_check` on several fixed gradient lists and scales
(power-of-two), some entirely finite, some with an injected `inf` or `nan`
in one array. For each case it independently computes the reference `skip`
flag with `np.isfinite` and the reference unscaled arrays with plain
elementwise division, then compares:

- your `skip` boolean against the reference boolean, and
- your unscaled arrays against the reference arrays via exact byte equality
  (`array.tobytes()` comparison, which correctly treats matching `inf`/`nan`
  bit patterns as equal without the usual `nan != nan` pitfall).

All cases must match exactly for `exact_match` to be `1.0`; any mismatch —
wrong skip decision, a rounded/approximate division, or returning the
original scaled values unmodified — makes it `0.0`. The gate requires
`exact_match == 1.0`.
