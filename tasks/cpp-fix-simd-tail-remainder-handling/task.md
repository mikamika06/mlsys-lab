## Context

Explicit vectorization using SIMD (ARM NEON here) processes multiple data elements in one instruction. A typical pattern loads 128-bit ($4 \times$ `int32_t`) lanes, accumulates, and reduces. A common bug is dropping the *tail* elements when the array length is not a multiple of the lane width.

## Task

`solve.cpp` contains a *broken* `long long simd_sum(const int* data, int n)` that sums an `int32` array using NEON, but only processes whole 4-wide chunks: it **silently ignores the remainder** whenever `n % 4 != 0`. Fix it by adding a scalar loop that adds the leftover elements the NEON loop skipped.

- Do not change the signature.
- The function must handle every length from `0` upward, including `n < 4`.

## Example

`simd_sum([1,2,3,4,5,6])` must return $21$ (the broken version stops after summing the first 4 elements, returning $10$). `simd_sum([])` must return $0$.

## What the gate checks

`main.cpp` calls `simd_sum` over eight fixed test arrays — including empty, single-element, and several lengths not divisible by $4$ — and prints each result. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's stdout. A NEON loop that never sums its dropped tail elements matches only on inputs whose length happens to be a multiple of $4$, and fails on everything else.
