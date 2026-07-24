#include "sol.hpp"

// Reference classifier: applies the C++20 undefined-behavior rules category
// by category. Returns 1 for UB, 0 for well-defined.
int classify_ub(const Snippet& s) {
    switch (s.op) {
        case SIGNED_ADD: {
            // Signed arithmetic overflow is UB. The value fits iff it lies
            // within the signed range of a `width`-bit type.
            long long mn = -(1LL << (s.width - 1));
            long long mx =  (1LL << (s.width - 1)) - 1;
            long long r  = s.a + s.b;
            return (r < mn || r > mx) ? 1 : 0;
        }
        case UNSIGNED_ADD:
            // Unsigned arithmetic wraps modulo 2^width; never UB.
            return 0;
        case ARRAY_IDX:
            // Reading outside [0, length) is UB.
            return (s.b < 0 || s.b >= s.a) ? 1 : 0;
        case UNINIT_READ:
            // Reading an uninitialized automatic (non-char) object is UB.
            return (s.flag == 0) ? 1 : 0;
        case NULL_DEREF:
            // Dereferencing a null pointer is UB.
            return (s.flag == 1) ? 1 : 0;
        case SHIFT:
            // In C++20 `E1 << E2` is UB iff E2 < 0 or E2 >= width. A negative
            // left operand is well-defined (result taken modulo 2^width).
            return (s.b < 0 || s.b >= s.width) ? 1 : 0;
        case TYPE_PUN:
            // A reinterpret_cast between incompatible types violates strict
            // aliasing (UB); memcpy / char access is well-defined.
            return (s.flag == 0) ? 1 : 0;
        default:
            return 0;
    }
}
