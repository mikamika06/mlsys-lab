#pragma once

// main.cpp compiles 8 fixed loops (listed below, verbatim) with
// clang++ -O2 and determines, for real, whether each one got autovectorized
// -- by disassembling ITS OWN compiled binary (`otool -tV` on argv[0]) and
// checking whether that function's machine code contains real NEON vector
// instructions (register suffixes like .4s / .2d). This is not a rule-based
// simulation: it is the actual answer the actual compiler gave, read back
// out of the actual object code.
//
// predictLoopN() must return YOUR guess -- true if you believe loop N gets
// autovectorized at -O2 on this compiler, false if you believe it stays
// scalar. main.cpp compares your 8 guesses against the real, freshly
// disassembled ground truth.
//
//   Loop 1 (vectorizable_add): out[i] = a[i] + b[i], plain float pointers
//     (no __restrict__). The vectorizer can still vectorize this by
//     inserting a RUNTIME alias check with a scalar fallback path.
//
//   Loop 2 (restrict_mul): out[i] = a[i] * 2.0f, with __restrict__ pointers
//     (no aliasing possible at all, nothing to check).
//
//   Loop 3 (int_add): out[i] = a[i] + b[i], over int32 arrays. Integer
//     addition has no reassociation/rounding concern.
//
//   Loop 4 (reduction_dep): s += a[i] / (s + 1.0f) -- s is read AND written
//     every iteration by an operation that depends on its own previous
//     value: a genuine loop-carried dependency chain.
//
//   Loop 5 (plain_sum_reduction): s += a[i], a plain floating-point sum.
//     Reordering float additions can change the rounded result, so without
//     relaxed-precision flags the compiler will not silently reassociate it.
//
//   Loop 6 (early_exit_loop): the loop body can `break` out early, based on
//     a value read from the array -- the trip count is not known up front.
//
//   Loop 7 (calls_opaque_fn): the loop body calls an external function the
//     compiler has no visibility into (declared but not defined here) --
//     it cannot prove the call has no side effects that depend on order.
//
//   Loop 8 (max_reduction): m = a[i] if a[i] > m, a running max via
//     compare-and-select. No reassociation-precision issue, but this
//     particular idiom does not get vectorized at -O2 on this compiler.
bool predictLoop1();
bool predictLoop2();
bool predictLoop3();
bool predictLoop4();
bool predictLoop5();
bool predictLoop6();
bool predictLoop7();
bool predictLoop8();
