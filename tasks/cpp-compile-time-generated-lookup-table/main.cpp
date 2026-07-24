#include <cstdio>
#include <cstdint>
#include "sol.hpp"

// FIXED driver. Do not edit. Generates a fixed n-entry LUT into a
// sentinel-filled buffer sized to the real n * sizeof(LutEntry), then
// prints the entry size and every output byte as two-digit hex.
int main() {
    const int n = 7;
    const int entry_size = (int)sizeof(LutEntry);
    const int out_len = n * entry_size;

    uint8_t* out = new uint8_t[out_len];
    for (int i = 0; i < out_len; ++i) out[i] = 0xFF;  // sentinel

    generate_lut_bytes(n, out, out_len);

    printf("entry_size=%d n=%d\n", entry_size, n);
    for (int i = 0; i < out_len; ++i) printf("%02x", out[i]);
    printf("\n");

    delete[] out;
    return 0;
}
