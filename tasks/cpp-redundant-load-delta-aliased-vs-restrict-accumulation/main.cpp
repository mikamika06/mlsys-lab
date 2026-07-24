#include <cstdio>
#include <vector>
#include "sol.hpp"

long g_load_count = 0;
long g_store_count = 0;

double load_double(const double* p) {
    ++g_load_count;
    return *p;
}
void store_double(double* p, double v) {
    ++g_store_count;
    *p = v;
}

// PROVIDED. Deterministic value generator (no rand(), no clock).
static double detval(int i) {
    unsigned x = (unsigned)(i * 2654435761u + 11u);
    x ^= x >> 13; x *= 2246822519u; x ^= x >> 16;
    return ((double)(x % 2000) / 100.0) - 10.0;
}

// FIXED driver. Do not edit. For each fixed n, resets the counters and
// runs accumulate_aliased and accumulate_hoisted independently (each
// starting from the same dest value, over the same src array), printing
// the observed load/store counts and the accumulated result for both.
int main() {
    int ns[] = {10, 37, 100};

    for (int n : ns) {
        std::vector<double> src((size_t)n);
        for (int i = 0; i < n; ++i) src[(size_t)i] = detval(i + n * 1000);
        double dest0 = 3.5;

        double dest_a = dest0;
        g_load_count = 0;
        g_store_count = 0;
        accumulate_aliased(&dest_a, src.data(), n);
        long aliased_loads = g_load_count, aliased_stores = g_store_count;

        double dest_h = dest0;
        g_load_count = 0;
        g_store_count = 0;
        accumulate_hoisted(&dest_h, src.data(), n);
        long hoisted_loads = g_load_count, hoisted_stores = g_store_count;

        printf("n=%d aliased_loads=%ld aliased_stores=%ld hoisted_loads=%ld "
               "hoisted_stores=%ld load_delta=%ld store_delta=%ld "
               "result_aliased=%.6f result_hoisted=%.6f\n",
               n, aliased_loads, aliased_stores, hoisted_loads, hoisted_stores,
               aliased_loads - hoisted_loads, aliased_stores - hoisted_stores,
               dest_a, dest_h);
    }
    return 0;
}
