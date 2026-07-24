#pragma once

// ---------------------------------------------------------------------------
// PROVIDED (do not change): instrumented scalar memory access. Every read
// or write of a float through these two functions bumps a real global
// counter.
// ---------------------------------------------------------------------------
extern long g_load_count;
extern long g_store_count;

float load_f(const float* p);
void  store_f(float* p, float v);

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS both. SAXPY: y[i] = (*a_ptr) * x[i] + y[i] for i in
// [0, n). Use load_f / store_f for EVERY access to *a_ptr and every x/y
// element (never dereference the raw pointers yourself), so the counters
// reflect exactly what your code does.
//
// saxpy_unhoisted: models the case where the compiler cannot prove
// a_ptr doesn't alias y, so it must reload *a_ptr on every iteration:
//     for i in [0,n): y[i] = load_f(a_ptr) * load_f(&x[i]) + load_f(&y[i])
//
// saxpy_hoisted: models the __restrict__-enabled case: a_ptr is proven
// not to alias x or y, so *a_ptr is loaded ONCE before the loop into a
// local scalar and reused every iteration (no per-iteration a_ptr load):
//     a = load_f(a_ptr)
//     for i in [0,n): y[i] = a * load_f(&x[i]) + load_f(&y[i])
// ---------------------------------------------------------------------------
void saxpy_unhoisted(const float* a_ptr, const float* x, float* y, int n);
void saxpy_hoisted(const float* a_ptr, const float* x, float* y, int n);
