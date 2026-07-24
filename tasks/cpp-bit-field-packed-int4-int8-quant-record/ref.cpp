#include "sol.hpp"
#include <cstring>

void pack_quant_block(int8_t scale, int32_t zero_point, const int weights[32],
                       uint8_t* out, int out_len) {
    QuantBlock qb{};
    qb.scale = scale;
    qb.zero_point = zero_point;
    for (int i = 0; i < 16; ++i) {
        uint8_t lo = (uint8_t)(weights[2 * i] & 0xF);
        uint8_t hi = (uint8_t)(weights[2 * i + 1] & 0xF);
        qb.weights[i] = (uint8_t)((hi << 4) | lo);
    }

    int n = out_len < (int)sizeof(QuantBlock) ? out_len : (int)sizeof(QuantBlock);
    std::memcpy(out, &qb, (size_t)n);
}
