#include "sol.hpp"
#include <cstring>

// BUG: reads from the start of each record instead of `field_offset`
// bytes into it, so the payload comes from whatever bytes happen to sit
// at the front of the struct instead of the actual double field.
void optimize_vector_loop(const unsigned char* buf, int n, int struct_size, int field_offset, double* out) {
    for (int i = 0; i < n; i++) {
        double v;
        std::memcpy(&v, buf + static_cast<long>(i) * struct_size, sizeof(double));
        out[i] = 2.0 * v;
    }
}
