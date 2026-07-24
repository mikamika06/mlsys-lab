#include "sol.hpp"

// TODO: decode the low `width` bits of `bits`.
//   - Mask `bits` down to `width` bits.
//   - If is_signed == 0, return that value directly.
//   - If is_signed != 0, use two's complement: when the top (bit width-1) is set,
//     the value is negative, equal to (masked value) - 2^width.
long long twos_complement_value(unsigned long long bits, int width, int is_signed) {
    // your code here
    return 0;
}
