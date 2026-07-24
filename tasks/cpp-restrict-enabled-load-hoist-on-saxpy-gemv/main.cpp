#include <cstdio>
#include <vector>
#include "sol.hpp"

long g_load_count = 0;
long g_store_count = 0;

float load_f(const float* p) {
    ++g_load_count;
    return *p;
}
void store_f(float* p, float v) {
    ++g_store_count;
    *p = v;
}

// PROVIDED. Deterministic value generator (no rand(), no clock).
static float detval(int i) {
    unsigned x = (unsigned)(i * 2654435761u + 3u);
    x ^= x >> 13; x *= 2246822519u; x ^= x >> 16;
    return ((float)(x % 2000) / 100.0f) - 10.0f;
}

// FIXED driver. Do not edit. For each fixed n, resets the counters and
// runs saxpy_unhoisted and saxpy_hoisted independently (each over its
// own y copy, same a/x), printing the observed load/store counts and the
// final y[0]/y[n-1] from both.
int main() {
    int ns[] = {10, 37, 100};

    for (int n : ns) {
        float a = 2.5f;
        std::vector<float> x((size_t)n);
        for (int i = 0; i < n; ++i) x[(size_t)i] = detval(i + n * 1000);

        std::vector<float> y_u((size_t)n), y_h((size_t)n);
        for (int i = 0; i < n; ++i) {
            float y0 = detval(i + n * 2000);
            y_u[(size_t)i] = y0;
            y_h[(size_t)i] = y0;
        }

        g_load_count = 0;
        g_store_count = 0;
        saxpy_unhoisted(&a, x.data(), y_u.data(), n);
        long u_loads = g_load_count, u_stores = g_store_count;

        g_load_count = 0;
        g_store_count = 0;
        saxpy_hoisted(&a, x.data(), y_h.data(), n);
        long h_loads = g_load_count, h_stores = g_store_count;

        printf("n=%d unhoisted_loads=%ld unhoisted_stores=%ld hoisted_loads=%ld "
               "hoisted_stores=%ld y_u0=%.6f y_uN=%.6f y_h0=%.6f y_hN=%.6f\n",
               n, u_loads, u_stores, h_loads, h_stores,
               y_u[0], y_u[(size_t)n - 1], y_h[0], y_h[(size_t)n - 1]);
    }
    return 0;
}
