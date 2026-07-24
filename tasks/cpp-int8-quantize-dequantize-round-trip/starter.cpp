#include "sol.hpp"

void quantize_dequantize(const float* data, int n, float scale, int zero_point, float* out) {
    // your code here
    for (int i = 0; i < n; i++) {
        out[i] = 0.0f;
    }
}
