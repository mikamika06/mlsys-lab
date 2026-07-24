## Context

When migrating neural network weights or activations to integer execution (e.g. `int8` Tensor Cores or SIMD instructions), real numbers get mapped to a bounded integer range. A linear asymmetric quantizer maps a float $x$ to a signed 8-bit integer $q \in [-128, 127]$ using a `scale` $S$ and `zero_point` $Z$:

$$q = \mathrm{clamp}(\mathrm{round}(x / S) + Z,\, -128,\, 127)$$

To measure the quantization error (or keep using the value in float logic), you **dequantize** back to an approximated float $\hat{x}$:

$$\hat{x} = (q - Z) \times S$$

The rounding must be round-half-to-even (banker's rounding — the IEEE-754 default, matching `std::nearbyint` under the default rounding mode), not round-half-away-from-zero (`std::round`), since those two disagree on exact `.5` boundaries.

## Task

Implement:

```cpp
void quantize_dequantize(const float* data, int n, float scale, int zero_point, float* out);
```

For each of the `n` floats in `data`: divide by `scale`, round half-to-even, add `zero_point`, clamp to `[-128, 127]`, store into a real `int8_t` (two's complement — this is where an unclamped value would silently wrap), then dequantize that `int8_t` back with `(q - zero_point) * scale` and write the result into `out[i]`.

## Example

For `x = 0.15`, `scale = 0.1`, `zero_point = 0`: `x / scale = 1.5`, which rounds half-to-even to `2`, giving `q = 2`. Dequantized: `(2 - 0) * 0.1 = 0.2`.

## What the gate checks

`main.cpp` generates 500 deterministic pseudo-random floats in `[-10, 10]` (a fixed-seed xorshift32 generator — no `rand()`, no clock), runs them through `quantize_dequantize` with `scale = 0.1`, `zero_point = 5`, and prints every reconstructed float. The candidate's numeric output is compared against the reference's with maximum absolute error (`max_abs_err <= 1e-5`). Rounding half-away-from-zero instead of half-to-even, or forgetting to clamp before narrowing to `int8_t`, both produce values that diverge from the reference on the affected inputs.
