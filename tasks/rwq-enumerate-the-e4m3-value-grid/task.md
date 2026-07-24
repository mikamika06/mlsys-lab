## Context

E4M3 is an 8-bit floating-point format: 1 sign bit `S`, 4 exponent bits `E`
(bias 7), 3 mantissa bits `M`, packed as `S EEEE MMM`. Its value rules are:

$$
v(S,E,M) =
\begin{cases}
\text{NaN} & E = 15,\ M = 7 \\[4pt]
(-1)^S \times 2^{\,E-7} \times \left(1+\dfrac{M}{8}\right) & E \neq 0,\ \text{not the NaN case} \\[8pt]
(-1)^S \times 2^{-6} \times \dfrac{M}{8} & E = 0
\end{cases}
$$

The `E=0` branch covers both the subnormal case (`M != 0`) and signed zero
(`M == 0`); it falls out of the same formula. The only reserved (non-finite)
encodings are the two patterns with `E=15, M=7` (one per sign) — every other
one of the 256 8-bit patterns decodes to a finite real number.

## Task

Implement `e4m3_value_grid()`:

```python
def e4m3_value_grid() -> dict:
    ...
```

Enumerate all 256 patterns (`S` in `{0,1}`, `E` in `0..15`, `M` in `0..7`),
decode each with the formula above, drop the two NaN patterns, and return a
dict with these keys:

- `"values"`: 1-D `float64` NumPy array of the **distinct** finite decoded
  values, sorted ascending (`+0.0` and `-0.0` count as the same value, `0.0`).
- `"n_finite"`: Python `int`, the number of distinct finite values (i.e.
  `len(values)`).
- `"max_finite"`: Python `float`, the largest finite value (`448.0`).
- `"min_subnormal"`: Python `float`, the smallest positive subnormal
  magnitude (the value at `E=0, M=1`).

No Python loops over 256 elements are required — build the grid with
vectorized NumPy (e.g. `np.arange`, bit shifts on an int array of codes
`0..255`, `np.unique`).

## Example

```python
grid = e4m3_value_grid()
grid["max_finite"]      # 448.0   (S=0, E=15, M=6: 2**8 * 1.75)
grid["min_subnormal"]   # 0.001953125   (S=0, E=0, M=1: 2**-6 * 1/8 == 2**-9)
grid["n_finite"]        # 253
grid["values"][0]       # -448.0  (most negative finite value)
grid["values"][126]     # 0.0     (the single zero entry)
```

## What the gate checks

The grader independently enumerates all 256 `S,E,M` combinations with a
vectorised NumPy oracle, applies the exact formula above, drops the two
`E=15, M=7` NaN codes, and deduplicates the remaining 254 values (the two
zero patterns collapse into one). This yields a canonical 253-element sorted
value array plus the three scalar constants.

`exact_match` is `1.0` only when your `"values"` array has the same length
and matches the oracle element-for-element, `"n_finite"` equals `253`, and
`"max_finite"` / `"min_subnormal"` equal the oracle's constants exactly
(these are all dyadic — exact powers of two and eighths — so a correct
implementation matches bit-for-bit; no floating-point tolerance is needed).
Any deviation (wrong bias, forgetting to drop the NaN codes, forgetting to
dedupe the two zeros, an off-by-one in the subnormal exponent) drops the
score to `0.0`.
