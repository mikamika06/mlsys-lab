## Context

The 8-bit **E4M3** float format (1 sign bit, 4 exponent bits with bias 7,
3 mantissa bits) has a limited dynamic range. Every real number, once you
look only at its magnitude, falls into one of four regimes relative to the
format's representable set:

$$
\text{MIN\_SUBNORMAL} = 2^{-9}, \qquad
\text{MIN\_NORMAL} = 2^{-6}, \qquad
\text{MAX\_NORMAL} = 448 .
$$

* **underflow-to-zero** — $|x| < \text{MIN\_SUBNORMAL}$: too small to be
  represented by even the smallest subnormal step; rounds to $0$. (Exact
  $0$ itself belongs here too — it stays $0$.)
* **subnormal** — $\text{MIN\_SUBNORMAL} \le |x| < \text{MIN\_NORMAL}$: in
  the denormalized range, where the exponent field is all-zero and there is
  no implicit leading `1` bit.
* **normal** — $\text{MIN\_NORMAL} \le |x| \le \text{MAX\_NORMAL}$: has a
  normal (implicit-`1`) exponent and fits without saturating.
* **overflow-clamped** — $|x| > \text{MAX\_NORMAL}$ (including $\pm\infty$
  and `nan`): too large to be represented; a real cast would clamp it to
  $\pm 448$.

## Task

Implement:

```python
def classify_e4m3_regime(x: list[float]) -> list[int]:
    ...
```

`x` is a list of any shape containing `float32`/`float64` values.
Return an `int64` array of the **same shape**, where each element is the
integer label of that value's regime:

| label | regime             |
|-------|--------------------|
| `0`   | underflow_to_zero  |
| `1`   | subnormal          |
| `2`   | normal             |
| `3`   | overflow_clamped   |

The classification is based purely on $|x|$ against the three boundaries
above — do not attempt to simulate rounding.

## Example

```python

x = [0.0, 1e-6, 0.01, 1.0, 448.0, 500.0, float('inf')]
classify_e4m3_regime(x)
# -> array([0, 0, 1, 2, 2, 3, 3])
#    0.0    -> underflow_to_zero (exact zero)
#    1e-6   -> underflow_to_zero (< 2^-9)
#    0.01   -> subnormal          (2^-9 <= 0.01 < 2^-6)
#    1.0    -> normal             (2^-6 <= 1.0 <= 448)
#    448.0  -> normal             (exactly the max)
#    500.0  -> overflow_clamped   (> 448)
#    inf    -> overflow_clamped
```

## What the gate checks

The grader loads a fixed fixture array `fp8_x.npy` — a mix of hand-picked
boundary values straddling every threshold (including `+/-0`, values just
below/above `MIN_SUBNORMAL`, `MIN_NORMAL`, and `MAX_NORMAL`, plus `+/-inf`)
and several thousand log-uniformly distributed random magnitudes with
random signs — and compares your labels element-by-element against a Python
oracle that applies the exact threshold logic above.

`exact_match` is `1.0` only if **every** label matches the oracle exactly
(same shape, same integer at every position), and `0.0` otherwise. Getting
a boundary comparison direction wrong (`<` vs `<=`), mixing up the
subnormal/normal cutoff, or not special-casing `nan`/`inf` as
overflow-clamped will mismatch on the corresponding fixture entries.
