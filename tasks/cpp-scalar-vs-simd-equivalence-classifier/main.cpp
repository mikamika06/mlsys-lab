#include <cstdio>
#include <cstring>
#include <cstdint>
#include "sol.hpp"

// A small seeded LCG -- fully deterministic, same sequence every run.
namespace {
    uint32_t seed = 0xC0FFEEu;
    float nextFloat(float scale) {
        seed = seed * 1664525u + 1013904223u;
        uint32_t bits = (seed >> 9) | 0x3F800000u;  // uniform in [1, 2)
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return (f - 1.5f) * scale;
    }
    int nextInt(int range) {
        seed = seed * 1664525u + 1013904223u;
        return (int)(seed % (uint32_t)range) - range / 2;
    }
}

// ---- scalar references (fixed, always correct, always left-to-right) ----

static void scalarSaxpy(float a, const float* x, const float* y, int n, float* out) {
    for (int i = 0; i < n; i++) out[i] = a * x[i] + y[i];
}

static float scalarFloatSum(const float* x, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += x[i];
    return acc;
}

static long long scalarIntSum(const int* x, int n) {
    long long acc = 0;
    for (int i = 0; i < n; i++) acc += x[i];
    return acc;
}

static void scalarFma(float a, const float* x, const float* y, int n, float* out) {
    for (int i = 0; i < n; i++) {
        volatile float prod = a * x[i];  // force a REAL separate rounding step here
        out[i] = prod + y[i];
    }
}

// ---------------------------------------------------------------------

int main() {
    const int N = 64;

    // Kernel 1: elementwise SAXPY
    {
        float a = nextFloat(10.0f);
        float x[N], y[N], outScalar[N], outSimd[N];
        for (int i = 0; i < N; i++) { x[i] = nextFloat(1000.0f); y[i] = nextFloat(1000.0f); }
        scalarSaxpy(a, x, y, N, outScalar);
        simdSaxpy(a, x, y, N, outSimd);
        int bitExact = (std::memcmp(outScalar, outSimd, sizeof(outScalar)) == 0) ? 1 : 0;
        printf("saxpy %d\n", bitExact);
    }

    // Kernel 2: float sum reduction
    {
        float x[N];
        for (int i = 0; i < N; i++) x[i] = nextFloat(50000.0f);
        float s1 = scalarFloatSum(x, N);
        float s2 = simdFloatSum(x, N);
        int bitExact = (std::memcmp(&s1, &s2, sizeof(float)) == 0) ? 1 : 0;
        printf("float_sum %d\n", bitExact);
    }

    // Kernel 3: int sum reduction
    {
        int x[N];
        for (int i = 0; i < N; i++) x[i] = nextInt(2000000);
        long long s1 = scalarIntSum(x, N);
        long long s2 = simdIntSum(x, N);
        int bitExact = (s1 == s2) ? 1 : 0;
        printf("int_sum %d\n", bitExact);
    }

    // Kernel 4: fused multiply-add vs separate mul+add
    {
        float a = nextFloat(1.0f);
        float x[N], y[N], outScalar[N], outSimd[N];
        for (int i = 0; i < N; i++) { x[i] = nextFloat(100000.0f); y[i] = nextFloat(100000.0f); }
        scalarFma(a, x, y, N, outScalar);
        simdFma(a, x, y, N, outSimd);
        int bitExact = (std::memcmp(outScalar, outSimd, sizeof(outScalar)) == 0) ? 1 : 0;
        printf("fma %d\n", bitExact);
    }

    return 0;
}
