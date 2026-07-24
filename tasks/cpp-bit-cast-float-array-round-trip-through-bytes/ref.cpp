#include "sol.hpp"
#include <bit>
#include <cstdint>

void floats_to_bytes(const float* x, int n, unsigned char* out) {
    for (int i = 0; i < n; i++) {
        std::uint32_t bits = std::bit_cast<std::uint32_t>(x[i]);
        out[4 * i + 0] = (unsigned char)(bits & 0xFFu);
        out[4 * i + 1] = (unsigned char)((bits >> 8) & 0xFFu);
        out[4 * i + 2] = (unsigned char)((bits >> 16) & 0xFFu);
        out[4 * i + 3] = (unsigned char)((bits >> 24) & 0xFFu);
    }
}

void bytes_to_floats(const unsigned char* in, int n, float* out) {
    for (int i = 0; i < n; i++) {
        std::uint32_t bits =
            (std::uint32_t)in[4 * i + 0] |
            ((std::uint32_t)in[4 * i + 1] << 8) |
            ((std::uint32_t)in[4 * i + 2] << 16) |
            ((std::uint32_t)in[4 * i + 3] << 24);
        out[i] = std::bit_cast<float>(bits);
    }
}
