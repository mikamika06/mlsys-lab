#include "sol.hpp"
#include <cstring>

void generate_lut_bytes(int n, uint8_t* out, int out_len) {
    const int esz = (int)sizeof(LutEntry);
    for (int i = 0; i < n; ++i) {
        LutEntry e{};
        e.index   = (char)i;
        e.doubled = (short)(i * 2);
        e.squared = i * i;

        int base = i * esz;
        int remaining = out_len - base;
        int n_copy = remaining < esz ? remaining : esz;
        if (n_copy > 0) std::memcpy(out + base, &e, (size_t)n_copy);
    }
}
