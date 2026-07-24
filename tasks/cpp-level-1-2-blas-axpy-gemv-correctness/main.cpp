#include <cstdio>
#include <dlfcn.h>
#include "sol.hpp"

void (*p_cblas_saxpy)(int, float, const float*, int, float*, int) = nullptr;
void (*p_cblas_sgemv)(int, int, int, int, float, const float*, int, const float*, int, float, float*, int) = nullptr;

static bool load_accelerate() {
    void* handle = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate", RTLD_NOW);
    if (!handle) return false;
    p_cblas_saxpy = reinterpret_cast<decltype(p_cblas_saxpy)>(dlsym(handle, "cblas_saxpy"));
    p_cblas_sgemv = reinterpret_cast<decltype(p_cblas_sgemv)>(dlsym(handle, "cblas_sgemv"));
    return p_cblas_saxpy != nullptr && p_cblas_sgemv != nullptr;
}

// FIXED driver.
int main() {
    if (!load_accelerate()) {
        printf("ACCELERATE_LOAD_FAILED\n");
        return 1;
    }

    // Level-1 AXPY: n=3 logical elements, x strided by 2 (interleaved with
    // junk values that must NOT be touched or read as data), y contiguous.
    const int n = 3, incx = 2, incy = 1;
    float x_axpy[6] = {1.0f, -9.0f, 2.0f, -9.0f, 3.0f, -9.0f};
    float y_axpy[3] = {10.0f, 20.0f, 30.0f};
    const float alpha_axpy = 2.5f;

    // Level-2 GEMV: A is 3x4 row-major, x_gemv length 4, y_gemv length 3.
    const int m = 3, gemv_n = 4;
    float A[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float x_gemv[4] = {1, 0, -1, 2};
    float y_gemv[3] = {100, 200, 300};
    const float alpha_gemv = 1.5f, beta_gemv = 0.5f;

    compute_axpy_gemv(n, alpha_axpy, x_axpy, incx, y_axpy, incy,
                       m, gemv_n, alpha_gemv, A, x_gemv, beta_gemv, y_gemv);

    for (int i = 0; i < n; i++) printf("%.6f\n", y_axpy[i]);
    for (int i = 0; i < m; i++) printf("%.6f\n", y_gemv[i]);
    return 0;
}
