## Context

Post-training int8 quantization needs a **clipping threshold** $t$: values are
mapped to signed integers with
$$
\hat{x} = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{x}{s}\right), -L, L\right) \cdot s,
\qquad s = \frac{t}{L}, \qquad L = 2^{\text{bits}-1}-1 .
$$
The choice of $t$ trades off two errors: too small clips too much signal, too
large wastes quantization levels on rare outliers. Production libraries
(e.g. TensorRT's `IInt8Calibrator` family) ship several calibrators that pick
$t$ differently:

* **min/max** — $t = \max_i |x_i|$. No clipping, but a single huge outlier
  blows up the scale and starves every normal value of resolution.
* **percentile** — $t = P_{99.99}(|x|)$, the 99.99th percentile of $|x|$.
  Clips the extreme tail on purpose to keep resolution for the bulk.
* **entropy** (KL-divergence calibration) — chooses $t$ to minimize the
  information lost by quantization, measured as KL divergence between a
  histogram of the real data and a histogram of what quantization would
  produce.

## Task

Implement `calibrate_and_score`:

```python
def calibrate_and_score(X: np.ndarray, num_bits: int = 8) -> dict:
    ...
```

* `X` — a 1-D or N-D `float64` array of activation values (flatten before
  processing).
* `num_bits` — always `8` for grading; your code should still use it wherever
  a bit-width-dependent constant appears (don't hardcode `8`/`127`).

Return `{"minmax": mse, "percentile": mse, "entropy": mse}`, where each `mse`
is `mean((X - dequant(quant(X, t)))**2)` (symmetric linear quantization as
above) using that method's threshold $t$.

**min/max threshold**: $t = \max(|X|)$.

**percentile threshold**: $t = $ the 99.99th percentile of $|X|$
(`np.percentile(np.abs(X), 99.99)`).

**entropy threshold** (must match exactly, including the constants below):

1. Let `num_quant_bins = 2**(num_bits-1)` (128 for 8-bit) and
   `num_bins = 16 * num_quant_bins` (2048 for 8-bit).
2. Build a histogram of $|X|$ with `num_bins` equal-width bins over
   `[0, max(|X|)]` (`np.histogram(x_abs, bins=num_bins, range=(0, x_abs.max()))`).
3. For each candidate prefix length `i` in
   `range(num_quant_bins, num_bins + 1, num_quant_bins)` (i.e. every multiple
   of `num_quant_bins` up to `num_bins`):
   - `P` = the first `i` histogram counts, with all counts beyond index `i`
     (the "outliers") added onto `P[-1]`.
   - Split `P` into `num_quant_bins` equal-size contiguous groups. Build `Q`
     (length `i`) by, within each group, spreading that group's total count
     evenly across only the group's *originally nonzero* bins (bins that were
     zero in `P` stay zero in `Q`; if a group sums to zero, its `Q` entries
     stay zero).
   - Normalize `P` and `Q` to sum to 1, replacing zero entries with `1e-8`
     before renormalizing (so `log` is always defined).
   - `KL = sum(P * log(P / Q))`.
4. Pick the `i` with the smallest `KL`. The threshold is `i * bin_width`
   where `bin_width` is the histogram's bin width.

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
X = rng.standard_t(2.0, size=5000)   # heavy-tailed
scores = calibrate_and_score(X, num_bits=8)
# scores == {"minmax": ..., "percentile": ..., "entropy": ...}
# min/max is typically far worse than the other two because a rare
# large sample from the t-distribution's tail dominates its threshold.
```

## What the gate checks

The grader builds three seeded heavy-tailed activations (Student's t with
different degrees of freedom and sizes) and computes the reference scores
with the same procedure described above.

* **mse** — mean squared difference between your three returned values and
  the reference's three values, pooled across all cases (`<= 1e-6`).
* **argmin_match** — the method with the smallest MSE in your returned dict
  must be the same method that has the smallest MSE in the reference, on
  every case (`== 1.0`).
