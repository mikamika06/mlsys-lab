#pragma once

// ---------------------------------------------------------------------------
// PROVIDED (defined in main.cpp): pointers to the REAL Apple Accelerate
// CBLAS routines, resolved via dlopen/dlsym so this task builds without any
// extra linker flags. Call through these exactly as you would call the
// real cblas_saxpy / cblas_sgemv.
// ---------------------------------------------------------------------------
extern void (*p_cblas_saxpy)(int n, float alpha, const float* x, int incx, float* y, int incy);
extern void (*p_cblas_sgemv)(int order, int trans, int m, int n, float alpha,
                              const float* a, int lda, const float* x, int incx,
                              float beta, float* y, int incy);

// Standard CBLAS constants (the values Accelerate's cblas_* expect).
constexpr int CBLAS_ROW_MAJOR = 101;
constexpr int CBLAS_NO_TRANS = 111;

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS, by calling the two BLAS routines above -- no hand
// loops.
//
// 1. Level-1 AXPY (strided): for i in [0, n),
//        y_axpy[i * incy] = alpha_axpy * x_axpy[i * incx] + y_axpy[i * incy]
//    via exactly ONE call to p_cblas_saxpy with the given n/alpha/incx/incy.
//
// 2. Level-2 GEMV: y_gemv = alpha_gemv * A * x_gemv + beta_gemv * y_gemv,
//    where A is (m x gemv_n), stored ROW-MAJOR with leading dimension
//    gemv_n, x_gemv has length gemv_n, y_gemv has length m. Via exactly ONE
//    call to p_cblas_sgemv with CBLAS_ROW_MAJOR / CBLAS_NO_TRANS.
//
// Both results are written back into y_axpy / y_gemv in place.
// ---------------------------------------------------------------------------
void compute_axpy_gemv(int n, float alpha_axpy, const float* x_axpy, int incx, float* y_axpy, int incy,
                        int m, int gemv_n, float alpha_gemv, const float* A, const float* x_gemv,
                        float beta_gemv, float* y_gemv);
