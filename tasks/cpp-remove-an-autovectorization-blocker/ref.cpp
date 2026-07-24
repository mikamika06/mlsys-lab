#include "sol.hpp"
#include <cstring>

void optimize_vector_loop(const unsigned char* buf, int n, int struct_size, int field_offset, double* out) {
    for (int i = 0; i < n; i++) {
        double v;
        std::memcpy(&v, buf + static_cast<long>(i) * struct_size + field_offset, sizeof(double));
        out[i] = 2.0 * v;
    }
}
