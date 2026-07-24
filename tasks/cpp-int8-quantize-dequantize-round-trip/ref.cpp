#include "sol.hpp"
#include <cmath>
#include <cstdint>

void quantize_dequantize(const float* data, int n, float scale, int zero_point, float* out) {
    for (int i = 0; i < n; i++) {
        double scaled = static_cast<double>(data[i]) / static_cast<double>(scale);
        long q = static_cast<long>(std::nearbyint(scaled)) + zero_point; // round-half-to-even
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        int8_t q8 = static_cast<int8_t>(q);
        out[i] = static_cast<float>((static_cast<int>(q8) - zero_point)) * scale;
    }
}
