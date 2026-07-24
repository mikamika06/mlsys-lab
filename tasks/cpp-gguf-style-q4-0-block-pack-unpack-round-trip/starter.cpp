#include "sol.hpp"

// TODO: implement both. See sol.hpp for the exact Q4_0 pack/unpack
// contract (scale computation, clamped rounding, nibble packing order).
void pack_q4_0(const float weights[32], uint8_t* out, int out_len) {
    (void)weights;
    (void)out;
    (void)out_len;
}

void unpack_q4_0(const uint8_t* block, int block_len, float out_weights[32]) {
    (void)block;
    (void)block_len;
    for (int i = 0; i < 32; ++i) out_weights[i] = 0.0f;
}
