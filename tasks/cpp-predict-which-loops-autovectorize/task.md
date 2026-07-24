## Context

The compiler's loop vectorizer rewrites a scalar loop to process several elements per iteration using SIMD (NEON) instructions -- but only when it can *prove* that's safe. Common things that block it:

- a genuine **loop-carried dependency** (each iteration needs the previous iteration's result for something other than a recognized reduction idiom),
- **floating-point reduction** without relaxed-precision flags (reordering float additions can change the rounded result, so the compiler won't silently do it),
- a **data-dependent early exit** (the trip count isn't known up front),
- a call to an **opaque function** the compiler can't see into (it can't prove the call has no order-dependent side effects).

Plain pointer aliasing, on the other hand, does *not* always block it: the vectorizer can insert a runtime alias check and fall back to a scalar loop only if the arrays actually overlap.

## Task

`main.cpp` compiles 8 fixed loops (their source is documented in full in `sol.hpp`) with `clang++ -O2` and disassembles its **own** compiled binary with `otool -tV` to determine, for real, which ones actually got vectorized. Implement `predictLoop1()` through `predictLoop8()` in `solve.cpp`: your best guess, `true` or `false`, for each loop.

The shipped guesses get three loops wrong: they assume a plain float sum reduction, a loop that calls an opaque external function, and a float max-reduction idiom all vectorize -- none of them do, for the reasons above (the max-reduction case specifically doesn't at `-O2` on this compiler, even though it has no reassociation-safety issue).

## Example

```cpp
bool predictLoop1() { return true; }   // out[i] = a[i] + b[i]: vectorizes
bool predictLoop4() { return false; }  // s += a[i] / (s + 1.0f): loop-carried, does not
```

## What the gate checks

`main.cpp` prints, for each loop, your prediction alongside the real disassembly-derived answer. The "real answer" column is identical no matter which file is linked (it's read straight off `main.cpp`'s own compiled code), so only your predictions can differ. Your printed output is compared against `ref.cpp`, compiled and run the same way: `max_abs_err <= 1e-9`.
