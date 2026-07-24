## Context

Two 8-bit floating-point formats are standard in production LLM inference for
quantizing KV-cache and activation tensors:

- **E4M3** — 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits. Finite
  range only (no infinities), max magnitude $448$, smallest positive
  (subnormal) magnitude $2^{-6}\cdot\frac18 = 2^{-9}$.
- **E5M2** — 1 sign bit, 5 exponent bits (bias 15), 2 mantissa bits. Max
  magnitude $57344$, smallest positive (subnormal) magnitude
  $2^{-14}\cdot\frac14 = 2^{-16}$.

Both formats round a real value $v$ to the nearest representable grid point.
For a normal value with exponent $e$, the two adjacent representable values
are spaced $2^{e}/2^{m}$ apart, where $m$ is the number of mantissa bits.
E4M3's extra mantissa bit ($m=3$ vs $m=2$) halves that spacing, so **at the
same exponent E4M3 is always more precise** — but its 4-bit exponent field
gives it roughly $2^{7}\times$ less *dynamic range* than E5M2's 5-bit
exponent field.

This produces a real, observable trade-off. Given a tensor of raw
(un-scaled) values:

- If every value comfortably fits inside both formats' range, E4M3's extra
  mantissa bit wins — smaller reconstruction error.
- If even one raw value exceeds E4M3's max magnitude of $448$ (routine for
  attention-sink / massive-activation outliers in real transformers), E4M3
  saturates that element at $\pm448$, producing a huge absolute error, while
  E5M2 (max $57344$) still represents it as a normal number with `mantissa`
  bits of precision — E5M2 wins instead.

## Task

Implement `fp8_format_errors`:

```python
def fp8_format_errors(x: np.ndarray) -> tuple[float, float]:
    ...
```

For the input tensor `x` (any shape, `float64`), quantize-then-dequantize it
through **both** formats, with **no rescaling** — round every element to the
nearest representable grid value of the format, saturating (clamping) at the
format's finite max magnitude ($448$ for E4M3, $57344$ for E5M2). Return

```
(e4m3_max_abs_err, e5m2_max_abs_err)
```

where each entry is $\max_i \lvert \hat{x}_i - x_i \rvert$ — the worst-case
absolute reconstruction error over all elements, for that format's
round-trip $\hat{x} = \text{dequant}(\text{quant}(x))$.

Rounding must be to the *nearest* representable value (break ties however
you like — the grader's test data essentially never lands exactly on a tie).
Subnormals must be handled: for exponent field $0$ the value is
$\text{sign}\cdot 2^{e_{\min}}\cdot(\text{mantissa}/2^m)$ (no implicit
leading 1); for exponent field $\geq 1$ it is
$\text{sign}\cdot 2^{(\text{exp}-\text{bias})}\cdot(1+\text{mantissa}/2^m)$.

## Example

```python
import numpy as np

x = np.array([1.0, 500.0, 0.002])
e4m3_err, e5m2_err = fp8_format_errors(x)
# 500.0 exceeds E4M3's 448 max -> it saturates to 448.0 -> abs error 52.0
# dominates e4m3_err. E5M2 represents ~500 as a normal number with only
# 2 mantissa bits, so its error there is much smaller.
```

## What the gate checks

The grader builds two deterministic KV-like tensors with `numpy`'s
`default_rng(7)`:

- a **uniform / well-scaled** tensor (`std ≈ 3`, nothing near either format's
  range limit),
- an **outlier-heavy** tensor (same bulk, plus a few raw values in
  `[600, 2000]` — beyond E4M3's 448 max).

For each tensor it computes the true `(e4m3_max_abs_err, e5m2_max_abs_err)`
pair from a from-scratch reference implementation of the E4M3/E5M2 minifloat
grids (built directly from the IEEE-754-style exponent/mantissa/bias
definitions above — no library shortcuts), and compares it to your
function's output:

- `uniform_err_diff` / `outlier_err_diff` — the largest absolute difference
  between your two returned errors and the oracle's, per tensor; must be
  `<= 1e-6`.
- `order_uniform` — on the uniform tensor your `e4m3` error must be strictly
  lower than your `e5m2` error.
- `order_outlier` — on the outlier tensor your `e5m2` error must be strictly
  lower than your `e4m3` error.

All four gates must pass. Getting the numeric reconstruction exactly right
(bit-exact rounding on both formats, including saturation) is what makes the
ordering come out correctly on its own — the ordering gates simply confirm
you reproduced the real trade-off, not just plausible-looking numbers.
