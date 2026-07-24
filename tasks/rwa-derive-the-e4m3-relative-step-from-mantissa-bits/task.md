## Context

A normalized binary floating-point value with $m$ mantissa bits represents
its significand as
$$
v = \pm(1+f)\cdot 2^{e}, \qquad f \in \{0, q, 2q, \dots, (2^m-1)q\}, \qquad q = 2^{-m},
$$
i.e. the fractional part $f$ only takes $2^m$ evenly-spaced values $q$
apart, called the *mantissa quantum*. Any real value that doesn't land
exactly on one of these grid points gets rounded to its nearest neighbor,
with a maximum absolute rounding error of half the quantum, $q/2$.

Because $f \in [0,1)$, the *relative* error of that rounding
($|\text{error}|/|v|$) is largest when $v$ is smallest within its binade
— i.e. near $f=0$ — where the denominator $(1+f)\cdot 2^e$ is closest to
$2^e$. The classic textbook bound for the worst-case relative rounding
error of round-to-nearest is therefore
$$
\varepsilon_{\text{rel}} \approx \frac{q/2}{1} = 2^{-(m+1)}.
$$

E4M3 has $m=3$ mantissa bits, so its analytic relative rounding step is
$2^{-4} = 0.0625$ — meaning individual E4M3-encoded values can be off from
the true value by up to roughly $6.25\%$, just from mantissa quantization
(ignoring subnormals and saturation, which are separate effects).

## Task

Implement `relative_rounding_step(values, mantissa_bits)`:

```python
def relative_rounding_step(values: np.ndarray, mantissa_bits: int) -> tuple[float, float]:
    ...
```

- `values`: a 1-D NumPy array of probe values, guaranteed to sit in the
  format's normal (non-subnormal, non-overflowing) representable range.
- `mantissa_bits`: the number of mantissa bits $m$ of the format being
  modeled (E4M3 uses $m=3$).

Return `(analytic_bound, empirical_max_rel_err)`:

1. `analytic_bound = 2 ** -(mantissa_bits + 1)` — the formula derived
   above.
2. `empirical_max_rel_err` — actually round every value in `values` to
   `mantissa_bits` mantissa bits and measure the max relative error:
   - Extract each value's normalized fractional mantissa: write
     $v = \pm(1+f)\cdot 2^{e}$ with $f\in[0,1)$ (e.g. via `np.frexp`,
     which gives a mantissa in $[0.5,1)$ — rescale it into $[1,2)$ form).
   - Round $f$ to the nearest multiple of $q = 2^{-\text{mantissa\_bits}}$.
   - Reconstruct the rounded value $\hat v = \pm(1+f_{\text{rounded}})\cdot 2^{e}$.
   - Compute $|\hat v - v| / |v|$ for every value, and return the max.

No hardcoded table of representable values is needed or wanted — round by
mantissa arithmetic directly, so the same code works for any
`mantissa_bits`.

## Example

```python
import numpy as np

# A value placed exactly halfway between two adjacent 3-bit-mantissa grid
# points, right at the low end of its binade -- the worst case.
q = 2.0 ** -3
values = np.array([1.0 + q / 2.0])   # 1.0625

analytic, empirical = relative_rounding_step(values, mantissa_bits=3)
print(analytic)    # 0.0625
print(empirical)   # ~0.0588  (q/2 / 1.0625)
```

## What the gate checks

The oracle tests `mantissa_bits` in `{3, 4, 5}` (E4M3's own width plus two
neighbors, so a solution can't just memorize the single constant
`0.0625` — it has to implement the general rounding-step derivation). For
each width it builds a *mid-range probe tensor*: several exponents in a
band well away from subnormal/overflow edges, each holding a value placed
exactly at the worst-case rounding midpoint near the low end of its binade
(where relative error is largest).

Your `(analytic_bound, empirical_max_rel_err)` must both match the
oracle's own values closely (this rules out returning two matching but
fabricated numbers). The graded metric, `rel_err`, is then
$$
\text{rel\_err} = \frac{|\text{empirical\_max\_rel\_err} - \text{analytic\_bound}|}{\text{analytic\_bound}},
$$
taken as the worst case over the three mantissa widths, and must be
`< 0.1` — i.e. your empirically-measured worst-case rounding error must
track the analytic $2^{-(m+1)}$ formula to within 10%. Rounding to the
wrong quantum, extracting the mantissa incorrectly, or returning the
absolute (not relative) error will throw the measured value far outside
that 10% band.
