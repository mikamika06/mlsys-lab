#include "sol.hpp"

// TODO: call p_cblas_saxpy for the strided AXPY, then p_cblas_sgemv (row
// major, no-transpose) for the GEMV. See sol.hpp for the exact contract.
void compute_axpy_gemv(int n, float alpha_axpy, const float* x_axpy, int incx, float* y_axpy, int incy,
                        int m, int gemv_n, float alpha_gemv, const float* A, const float* x_gemv,
                        float beta_gemv, float* y_gemv) {
    // your code here
}
