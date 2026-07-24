#include "sol.hpp"

// BUG: does the arithmetic directly in int32_t, which signed-overflows
// (undefined behavior) instead of saturating.
void saturating_arithmetic(const int32_t* a, const int32_t* b, int n, Op op, int32_t* out) {
    for (int i = 0; i < n; i++) {
        int32_t r;
        switch (op) {
            case Op::Add: r = a[i] + b[i]; break;
            case Op::Sub: r = a[i] - b[i]; break;
            case Op::Mul: r = a[i] * b[i]; break;
            default: r = 0; break;
        }
        out[i] = r;
    }
}
