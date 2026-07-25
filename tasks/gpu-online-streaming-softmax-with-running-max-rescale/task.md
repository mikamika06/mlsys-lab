## Context

Numerically-safe softmax subtracts the max before exponentiating:
$\text{softmax}(x)_i = \exp(x_i - m) / \sum_j \exp(x_j - m)$, where
$m = \max_j x_j$. That normally needs two full passes — one to find $m$,
one to sum the exponentials — which is exactly what a *streaming* kernel
(processing tiles as they arrive, flash-attention-style) can't afford:
the true max might not be known until the very last tile.

**Online softmax** gets away with a single streaming pass by tracking a
*running* max $m$ and a *running* sum of exponentials $l$, and correcting
$l$ every time $m$ changes. Every term already folded into $l$ was
computed relative to the *old* max, so rescale them all by
$\exp(m_{\text{old}} - m_{\text{new}})$ before adding the new term in:

$$m_{\text{new}} = \max(m, x_i) \qquad l \leftarrow l \cdot e^{m - m_{\text{new}}} + e^{x_i - m_{\text{new}}} \qquad m \leftarrow m_{\text{new}}$$

After the stream, `m` and `l` are *exactly* what a two-pass softmax would
have computed — the rescale makes the running sum retroactively correct
every time the max moves.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void online_softmax(float* out, const float* x, int n);
```

Single-threaded (`threadIdx.x == 0` only). First pass: `m = x[0]`,
`l = 1.0f`; for `i` in `[1, n)`, apply the update above. Second pass: for
`i` in `[0, n)`, `out[i] = expf(x[i] - m) / l`, using the *final* `m`
and `l` from the first pass.

## Example

Stream `2.0, 1.0, 3.0`: start `m=2, l=1`. Step `x=1`: `m` stays `2`
(`1<2`), `l = 1*e^0 + e^{1-2}` = `1 + e^{-1}`. Step `x=3`: `m_new=3`,
`l = (1+e^{-1})\cdot e^{2-3} + e^{3-3} = (1+e^{-1})e^{-1} + 1` — the
*entire* running sum gets rescaled by `e^{-1}` because the max just moved
from 2 to 3. Skip that rescale and the final sum is systematically too
large.

## What the gate checks

`max_abs_err <= 1e-6` against a standard two-pass numpy softmax, on a
fixed 40-value input with a late spike (`x[20] = 20.0`, the true max,
arriving after 20 smaller values) chosen specifically so the running max
updates mid-stream and a missing or wrong rescale term is forced to show
up in the final normalized output, not just in an intermediate value that
never gets checked.
