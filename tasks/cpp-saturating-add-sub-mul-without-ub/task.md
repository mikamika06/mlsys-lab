## Context

In C++, signed integer overflow (e.g. `INT32_MAX + 1`) is undefined behavior — not wraparound, not a trap, just UB, which lets an optimizing compiler assume it never happens and remove or misuse related checks. **Saturating** arithmetic instead clamps the mathematically exact result to the valid range:

$$\mathrm{clamp}(a \mathbin{op} b,\; \mathrm{INT32\_MIN},\; \mathrm{INT32\_MAX})$$

for `op` in `{+, -, *}`, computed without ever letting the `int32_t` arithmetic itself overflow.

## Task

Implement:

```cpp
void saturating_arithmetic(const int32_t* a, const int32_t* b, int n, Op op, int32_t* out);
```

For each `i`, compute `a[i] op b[i]` in a wider type (`int64_t`, where the true mathematical result of any `int32_t` pair always fits — even $\mathrm{INT32\_MAX} \times \mathrm{INT32\_MAX}$), clamp that to `[INT32_MIN, INT32_MAX]`, and narrow the clamped value into `out[i]`. `Op` is one of `Op::Add`, `Op::Sub`, `Op::Mul`.

## Example

`a = INT32_MAX`, `b = 100`, `op = Add`: the true sum $2147483647 + 100$ exceeds `INT32_MAX`, so the result saturates to `INT32_MAX`. `a = INT32_MIN`, `b = INT32_MAX`, `op = Sub`: the true difference is far below `INT32_MIN`, so the result saturates to `INT32_MIN`.

## What the gate checks

`main.cpp` runs a fixed array of 8 `int32_t` pairs — including exact-boundary values (`INT32_MAX`, `INT32_MIN`) chosen to overflow under every operator — through all three operations and prints every result. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's. Computing the arithmetic directly in `int32_t` instead of a wider type invokes real signed-overflow UB — on this compiler it manifests as silent two's-complement wraparound, producing numbers with the wrong sign instead of the correctly saturated bound.
