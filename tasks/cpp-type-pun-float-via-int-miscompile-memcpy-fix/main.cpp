#include <cstdio>
#include <cstdint>
#include "sol.hpp"

// Fixed, deterministic driver. It exercises the float<->bits round trip and the
// exponent-field power-of-two scale, then prints every numeric result.
int main() {
    // 1) float -> raw IEEE-754 bit pattern (printed as unsigned decimals)
    const int NF = 8;
    float fvals[NF] = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 1.5f, -0.25f, 3.14159265f};
    for (int i = 0; i < NF; i++) printf("%u ", float_to_bits(fvals[i]));
    printf("\n");

    // 2) raw IEEE-754 bit pattern -> float
    const int NB = 6;
    uint32_t bvals[NB] = {0x00000000u, 0x3F800000u, 0xBF800000u,
                          0x40490FDBu, 0x40000000u, 0xC2C80000u};
    for (int i = 0; i < NB; i++) printf("%.6f ", bits_to_float(bvals[i]));
    printf("\n");

    // 3) exponent-field power-of-two scaling: multiply all by 2^4 = 16
    const int NS = 8;
    float a[NS] = {1.0f, -2.0f, 3.5f, -0.5f, 8.0f, 100.0f, -0.25f, 6.0f};
    scale_pow2_inplace(a, NS, 4);
    double sa = 0;
    for (int i = 0; i < NS; i++) { printf("%.6f ", a[i]); sa += a[i]; }
    printf("\n");

    // ... and divide another array by 2^3 = 8 (negative exponent shift)
    float b[NS] = {16.0f, -32.0f, 64.0f, 8.0f, 128.0f, 1600.0f, -4.0f, 96.0f};
    scale_pow2_inplace(b, NS, -3);
    double sb = 0;
    for (int i = 0; i < NS; i++) { printf("%.6f ", b[i]); sb += b[i]; }
    printf("\nsumA=%.6f sumB=%.6f\n", sa, sb);
    return 0;
}
