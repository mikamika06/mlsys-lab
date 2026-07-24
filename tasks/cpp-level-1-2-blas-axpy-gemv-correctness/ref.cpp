#include "sol.hpp"

void compute_axpy_gemv(int n, float alpha_axpy, const float* x_axpy, int incx, float* y_axpy, int incy,
                        int m, int gemv_n, float alpha_gemv, const float* A, const float* x_gemv,
                        float beta_gemv, float* y_gemv) {
    p_cblas_saxpy(n, alpha_axpy, x_axpy, incx, y_axpy, incy);
    p_cblas_sgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, m, gemv_n, alpha_gemv, A, gemv_n,
                  x_gemv, 1, beta_gemv, y_gemv, 1);
}
