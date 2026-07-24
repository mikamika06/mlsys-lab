#include "sol.hpp"

void saxpy_unhoisted(const float* a_ptr, const float* x, float* y, int n) {
    for (int i = 0; i < n; ++i) {
        float a = load_f(a_ptr);
        float xv = load_f(&x[i]);
        float yv = load_f(&y[i]);
        store_f(&y[i], a * xv + yv);
    }
}

void saxpy_hoisted(const float* a_ptr, const float* x, float* y, int n) {
    float a = load_f(a_ptr);
    for (int i = 0; i < n; ++i) {
        float xv = load_f(&x[i]);
        float yv = load_f(&y[i]);
        store_f(&y[i], a * xv + yv);
    }
}
