#include "sol.hpp"
#include <cstdint>
#include <limits>

void saturating_arithmetic(const int32_t* a, const int32_t* b, int n, Op op, int32_t* out) {
    const int64_t lo = std::numeric_limits<int32_t>::min();
    const int64_t hi = std::numeric_limits<int32_t>::max();
    for (int i = 0; i < n; i++) {
        int64_t x = static_cast<int64_t>(a[i]);
        int64_t y = static_cast<int64_t>(b[i]);
        int64_t r;
        switch (op) {
            case Op::Add: r = x + y; break;
            case Op::Sub: r = x - y; break;
            case Op::Mul: r = x * y; break;
            default: r = 0; break;
        }
        if (r < lo) r = lo;
        if (r > hi) r = hi;
        out[i] = static_cast<int32_t>(r);
    }
}
