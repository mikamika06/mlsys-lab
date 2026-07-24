#include <cstdio>
#include <cstdint>
#include "sol.hpp"

// FIXED driver. Do not edit. Deterministic scale/zero_point/weights, calls
// the learner's packer into a sentinel-filled buffer sized to the real
// sizeof(QuantBlock), then prints the struct size and every output byte as
// two-digit hex.
int main() {
    int8_t  scale = 10;
    int32_t zero_point = -3;
    int weights[32];
    for (int i = 0; i < 32; ++i) weights[i] = (i * 5 + 3) % 16;

    uint8_t out[sizeof(QuantBlock)];
    for (size_t i = 0; i < sizeof(QuantBlock); ++i) out[i] = 0xFF;  // sentinel

    pack_quant_block(scale, zero_point, weights, out, (int)sizeof(QuantBlock));

    printf("size=%d\n", (int)sizeof(QuantBlock));
    for (size_t i = 0; i < sizeof(QuantBlock); ++i) printf("%02x", out[i]);
    printf("\n");
    return 0;
}
