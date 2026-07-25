#pragma once

// main.cpp compiles 5 fixed loops (listed below, verbatim) with clang++ -O2
// and determines, for real, whether each one got autovectorized -- by
// disassembling ITS OWN compiled binary (`otool -tV` on argv[0]) and
// checking whether that function's machine code contains a real NEON vector
// instruction (register suffix .4s / .2s / .2d / .16b / .8h). This is not a
// rule-based simulation: it is the actual answer the actual compiler gave,
// read back out of the actual object code.
//
// predictLoopN() must return YOUR guess -- true if you believe loop N gets
// autovectorized at -O2 on this compiler, false if you believe it stays
// scalar. main.cpp compares your 5 guesses against the real, freshly
// disassembled ground truth.
//
//   Loop 1 (elementwise_add): a[i] = b[i] + c[i], plain float pointers, no
//     loop-carried dependency, no aliasing hazard between distinct arrays.
//
//   Loop 2 (carried_dep): a[i] = a[i-1] + b[i] -- a[i] is read AND written
//     by a chain that depends on the immediately preceding iteration: a
//     genuine loop-carried dependency.
//
//   Loop 3 (plain_sum): s += a[i], a plain floating-point sum reduction.
//     Reordering float additions can change the rounded result, so without
//     relaxed-precision flags the compiler will not silently reassociate
//     it.
//
//   Loop 4 (branch_free_select): a[i] = b[i] > 0.0f ? b[i] : 0.0f -- a
//     branch that is really just a per-element max-with-zero, a pattern
//     compilers turn into a compare + select with no data dependency
//     between elements.
//
//   Loop 5 (nonuniform_index): a[i] = b[(i*i) % N] -- the index into b is a
//     non-uniform (quadratic) function of i, not a constant stride, so
//     consecutive iterations do not read consecutive memory.
bool predictLoop1();
bool predictLoop2();
bool predictLoop3();
bool predictLoop4();
bool predictLoop5();
