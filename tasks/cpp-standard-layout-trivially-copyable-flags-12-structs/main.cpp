#include <cstdio>
#include "sol.hpp"

// FIXED driver. Do not edit. Calls classify(), prints the 24-bit vector plus
// two summary counts. Deterministic: no input, no randomness.
int main() {
    int out[24];
    for (int i = 0; i < 24; i++) out[i] = -1;  // sentinel: an untouched cell shows up
    classify(out);

    for (int i = 0; i < 24; i++) printf("%d ", out[i]);
    printf("\n");

    int n_standard_layout = 0, n_trivially_copyable = 0;
    for (int k = 0; k < 12; k++) {
        n_standard_layout   += out[2 * k];
        n_trivially_copyable += out[2 * k + 1];
    }
    printf("standard_layout=%d trivially_copyable=%d\n",
           n_standard_layout, n_trivially_copyable);
    return 0;
}
