## Context

Horner's method evaluates a degree-$d$ polynomial with $d$ multiply-adds
instead of the $O(d^2)$ a naive term-by-term evaluation would need. Each
step's multiply-then-add,
`result = result * x + coeffs[i]`, can be done as a single **fused
multiply-add** (`std::fma`) instead of a separate multiply and add. FMA
rounds once instead of twice, so it is both more accurate and (on
hardware that supports it) a single instruction instead of two.

## Task

Implement

```cpp
void horner_eval(const double* coeffs, int n_coeffs, double x, double* value_out, long* fma_count_out);
```

With $d = \text{n\_coeffs} - 1$ and `coeffs[d]` the highest-degree
coefficient:

$$\text{result} = \text{coeffs}[d]; \quad \text{for } i = d-1 \text{ downto } 0: \ \text{result} = \mathrm{fma}(\text{result}, x, \text{coeffs}[i])$$

Use `std::fma` (from `<cmath>`) for every step — not a separate `*` and
`+`. Write the final value into `*value_out` and the number of `fma()`
calls made (exactly `d`) into `*fma_count_out`.

## Example

For coefficients `{2.0, -3.0, 0.5, 1.25, -0.75, 4.0}` (degree $d=5$,
6 coefficients) evaluated at `x = 1.37`, Horner's method makes exactly 5
`fma()` calls, one per coefficient below the leading one.

## What the gate checks

`exact_match`: the driver prints the evaluated value and the fma count
for one fixed degree-5 polynomial and `x`. Using `result * x +
coeffs[i]` instead of `std::fma` still calls `fma_count_out = 0` instead
of `5` and fails the match even where the value itself is close; a
starter that never writes through the output pointers leaves the
driver's sentinel value in place and fails outright.
