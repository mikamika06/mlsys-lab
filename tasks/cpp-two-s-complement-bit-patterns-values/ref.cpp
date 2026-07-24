#include "sol.hpp"

long long twos_complement_value(unsigned long long bits, int width, int is_signed) {
    // Mask off everything above the field width (width is 1..32, so this is safe).
    unsigned long long mask = (1ull << width) - 1ull;
    unsigned long long u = bits & mask;
    if (!is_signed) {
        return (long long)u;                 // plain unsigned magnitude
    }
    unsigned long long sign_bit = 1ull << (width - 1);
    if (u & sign_bit) {
        // Sign bit set: subtract 2^width to get the two's-complement value.
        return (long long)u - (long long)(mask + 1ull);
    }
    return (long long)u;
}
