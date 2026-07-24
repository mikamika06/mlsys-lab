## Context

Auto-vectorizing compilers (Clang/GCC at `-O2`/`-O3`) can turn a loop into parallel SIMD instructions only when each iteration's memory access can be computed directly from the loop index — no data-dependent offsets, no reading a value the previous iteration just wrote. A straight `for (i) out[i] = f(in[i]);` scan over fixed-stride records is exactly that shape.

Records here are `struct_size`-byte blocks packed back-to-back; each record's `double` payload lives at a fixed byte offset (`field_offset`) within it — computed the same way a real compiler computes `offsetof` under natural alignment (`char`=1, `short`=2, `int`/`float`=4, `long`/`double`/pointer=8, alignment equals size).

## Task

`solve.cpp` contains a *broken* `optimize_vector_loop` that reads each record's payload from the **wrong byte offset** — the start of the record instead of `field_offset` bytes into it. Fix it: for each of the `n` records, `memcpy` the 8-byte `double` at `buf + i * struct_size + field_offset`, multiply by `2.0`, and store it in `out[i]`.

- Do not change the signature.
- The address for record `i` must be computed directly from `i` — not from `out[i-1]` or any other previous iteration's result.

## Example

For `struct_size = 24`, `field_offset = 8`, record `0`'s payload is the 8 bytes at `buf[8..16)`; record `1`'s is at `buf[32..40)`. If that payload is `1.5`, `out[0]` must be `3.0`.

## What the gate checks

`main.cpp` builds four fixed byte buffers (mirroring four different struct layouts and record counts) with a known ascending `double` sequence written at each record's `field_offset`, and prints every `optimize_vector_loop` result. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's. Reading from the record's start instead of `field_offset` picks up whatever bytes precede the payload field (padding, or an unrelated field), producing numbers that have nothing to do with the actual data.
