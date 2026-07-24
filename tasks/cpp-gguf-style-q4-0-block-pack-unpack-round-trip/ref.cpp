#include "sol.hpp"
#include <cmath>
#include <cstring>

void pack_q4_0(const float weights[32], uint8_t* out, int out_len) {
    float max_abs = 0.0f;
    for (int i = 0; i < 32; ++i) {
        float a = std::fabs(weights[i]);
        if (a > max_abs) max_abs = a;
    }
    float d = max_abs / 7.0f;

    int8_t q[32];
    if (d == 0.0f) {
        for (int i = 0; i < 32; ++i) q[i] = 0;
    } else {
        for (int i = 0; i < 32; ++i) {
            float rounded = std::round(weights[i] / d);
            if (rounded < -8.0f) rounded = -8.0f;
            if (rounded > 7.0f) rounded = 7.0f;
            q[i] = (int8_t)rounded;
        }
    }

    block_q4_0 blk{};
    blk.d = encode_fp16(d);
    for (int i = 0; i < 16; ++i) {
        uint8_t lo = (uint8_t)((q[i] + 8) & 0x0F);
        uint8_t hi = (uint8_t)((q[i + 16] + 8) & 0x0F);
        blk.qs[i] = (uint8_t)(lo | (hi << 4));
    }

    int n = out_len < (int)sizeof(block_q4_0) ? out_len : (int)sizeof(block_q4_0);
    std::memcpy(out, &blk, (size_t)n);
}

void unpack_q4_0(const uint8_t* block, int block_len, float out_weights[32]) {
    block_q4_0 blk{};
    int n = block_len < (int)sizeof(block_q4_0) ? block_len : (int)sizeof(block_q4_0);
    std::memcpy(&blk, block, (size_t)n);

    float d = decode_fp16(blk.d);
    for (int i = 0; i < 16; ++i) {
        uint8_t b = blk.qs[i];
        int q0 = (int)(b & 0x0F) - 8;
        int q1 = (int)((b >> 4) & 0x0F) - 8;
        out_weights[i]      = (float)q0 * d;
        out_weights[i + 16] = (float)q1 * d;
    }
}
