#include <cstdio>
#include "sol.hpp"

// FIXED driver. n=13 is not a multiple of the 4-way unroll factor, so
// the last group has 13 % 4 = 1 leftover element.
int main() {
    const int n = 13;
    const float s = 2.5f;
    float in[n];
    for (int i = 0; i < n; i++) in[i] = (float)i;

    float out[n];
    for (int i = 0; i < n; i++) out[i] = -999.0f;  // sentinel

    scale_unrolled(in, n, s, out);

    for (int i = 0; i < n; i++) printf("%.2f ", out[i]);
    printf("\n");
    return 0;
}
