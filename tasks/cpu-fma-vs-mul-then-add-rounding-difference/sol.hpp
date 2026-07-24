#pragma once

// LEARNER IMPLEMENTS.
//
// fma_result: compute a*b + c with a single correctly-rounded step --
// the mathematically exact product a*b (infinite precision) is added to
// c and only THAT final sum is rounded to float, once. `std::fma` is
// defined by the C++ standard to be exactly this: a fused multiply-add
// with a single rounding.
float fma_result(float a, float b, float c);

// naive_result: compute a*b + c the "obvious" separate way: round the
// product to float FIRST (one rounding), then add c and round AGAIN (a
// second rounding). Store the intermediate product in a `volatile
// float` -- that forces the compiler to actually materialize (round) it
// before the addition runs, instead of silently fusing the two
// statements back into one hardware FMA instruction (which is legal
// within a single unbroken expression and would erase the very
// double-rounding effect this function exists to demonstrate).
float naive_result(float a, float b, float c);
