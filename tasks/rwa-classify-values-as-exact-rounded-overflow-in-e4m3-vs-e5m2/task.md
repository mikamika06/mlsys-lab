## Context

Every fp8 format represents only a finite set of magnitudes — the "grid" of
values its exponent/mantissa bits can spell out exactly, up to some largest
finite magnitude $M$. For any real value $v$, there are exactly three
possibilities relative to a given format:

- **exact** — $|v|$ is itself a grid point; the format stores it losslessly.
- **rounded** — $|v| \le M$ but $|v|$ is not a grid point; encoding it will
  move it to the nearest representable neighbor, changing its value while
  staying within range.
- **overflow** — $|v| > M$; the format's dynamic range simply doesn't reach
  that far, and encoding it saturates/loses far more than ordinary rounding.

The two common 8-bit floating formats used in ML inference have very
different ranges because they split their 8 bits differently:

| format | sign | exponent | mantissa | max finite $M$ |
|---|---|---|---|---|
| E4M3 | 1 | 4 | 3 | $448$ |
| E5M2 | 1 | 5 | 2 | $57344$ |

E4M3 trades range for precision (finer grid, smaller $M$); E5M2 trades
precision for range (coarser grid, larger $M$). The same value can land in
different buckets depending on which format you ask about — e.g. $500$ is
`overflow` for E4M3 but perfectly in-range for E5M2.

## Task

Implement `classify_fp8(values, grid, max_finite)`:

```python
def classify_fp8(values: np.ndarray, grid: np.ndarray, max_finite: float) -> np.ndarray:
    ...
```

- `values`: a NumPy array of values to classify.
- `grid`: a 1-D NumPy array of every representable **nonnegative** magnitude
  in the target format, in ascending order (`grid[-1] == max_finite`).
- `max_finite`: the largest finite magnitude the format can represent.

For every value, using $|v|$, return one of the three labels above as a
NumPy array of strings, same shape as `values`:

- `"exact"` if $|v|$ is exactly a member of `grid`,
- `"rounded"` if $|v| \le$ `max_finite` but $|v|$ is not a member of `grid`,
- `"overflow"` if $|v| >$ `max_finite`.

## Example

```python
import numpy as np

grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])   # toy 3-point-per-octave grid
classify_fp8(np.array([1.0, 1.3, 5.0]), grid, max_finite=2.0)
# -> array(["exact", "rounded", "overflow"])
```

## What the gate checks

The gate builds the *real* E4M3 and E5M2 grids from their bit-layout
formulas (sign/exponent/mantissa, decoding every representable code) — this
is the oracle, not a hardcoded list. It then classifies a shared batch of
test values under **both** formats: values picked to sit exactly on each
format's grid, values nudged slightly off a grid point (so they must round
but stay in range), and values chosen to overflow one format while staying
in range for the other (like the $500$ example above), plus a batch of
random values from a seeded generator.

Your labels are compared element-for-element against the reference labels
for each format; the metric is `1.0` only if every label matches for both
formats, else `0.0`. A solution that classifies purely by magnitude
threshold — calling anything below `max_finite` `"exact"` without checking
grid membership — will mislabel every genuinely rounded value as exact and
fail the gate.
