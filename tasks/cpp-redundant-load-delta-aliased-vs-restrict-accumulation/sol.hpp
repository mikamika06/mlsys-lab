#pragma once

// ---------------------------------------------------------------------------
// PROVIDED (do not change): instrumented scalar memory access. Every read
// or write of a double through these two functions bumps a real global
// counter, so the driver can report REAL load/store counts instead of a
// hand-computed formula.
// ---------------------------------------------------------------------------
extern long g_load_count;
extern long g_store_count;

double load_double(const double* p);
void   store_double(double* p, double v);

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS both loops. `dest` points at one accumulator cell,
// `src` is an array of `n` doubles to accumulate into it. You MUST use
// load_double / store_double for every access to *dest and every src
// element (never dereference the raw pointers yourself), so the counters
// reflect exactly what your code does.
//
// accumulate_aliased: models the PESSIMISTIC case where the compiler
// cannot prove `dest` doesn't alias an element of `src`, so it must
// reload *dest and store it back on every single iteration:
//     for i in [0, n): *dest = *dest + src[i]     (reload + store, EVERY i)
//
// accumulate_hoisted: models the OPTIMIZED case (dest marked `restrict`,
// or hoisted into a register by the compiler / by hand): *dest is loaded
// ONCE before the loop into a local accumulator, and stored back ONCE
// after the loop:
//     acc = *dest; for i in [0, n): acc += src[i]; *dest = acc;
// ---------------------------------------------------------------------------
void accumulate_aliased(double* dest, const double* src, int n);
void accumulate_hoisted(double* dest, const double* src, int n);
