#include <cstdio>
#include "sol.hpp"

// FIXED driver. Twenty canonical snippets spanning the UB taxonomy:
// signed overflow, out-of-bounds read, uninitialized read, null deref,
// oversized/negative shift, and strict-aliasing (type punning). Roughly half
// are genuine undefined behavior; the rest are look-alikes that are actually
// well-defined under C++20.
int main() {
    const Snippet snips[20] = {
        //  1  INT_MAX + 1 (32-bit signed)         -> signed overflow, UB
        { SIGNED_ADD,   2147483647LL,  1LL,  32, 0 },
        //  2  2000000000 + 100 (32-bit signed)    -> fits, defined
        { SIGNED_ADD,   2000000000LL,  100LL, 32, 0 },
        //  3  UINT_MAX + 1 (32-bit unsigned)      -> wraps modulo 2^32, defined
        { UNSIGNED_ADD, 4294967295LL,  1LL,  32, 0 },
        //  4  100 + 27 (8-bit signed) = 127       -> fits, defined
        { SIGNED_ADD,   100LL,         27LL,  8, 0 },
        //  5  100 + 28 (8-bit signed) = 128 > 127 -> signed overflow, UB
        { SIGNED_ADD,   100LL,         28LL,  8, 0 },
        //  6  a[10] on int a[10]                   -> out-of-bounds read, UB
        { ARRAY_IDX,    10LL,          10LL, 32, 0 },
        //  7  a[9]  on int a[10]                   -> last valid element, defined
        { ARRAY_IDX,    10LL,          9LL,  32, 0 },
        //  8  a[-1] on int a[10]                   -> out-of-bounds read, UB
        { ARRAY_IDX,    10LL,          -1LL, 32, 0 },
        //  9  read an uninitialized int            -> uninitialized read, UB
        { UNINIT_READ,  0LL,           0LL,  32, 0 },
        // 10  read the int after initializing it   -> defined
        { UNINIT_READ,  0LL,           0LL,  32, 1 },
        // 11  *p where p == nullptr                -> null dereference, UB
        { NULL_DEREF,   0LL,           0LL,  64, 1 },
        // 12  *p where p points at a live object   -> defined
        { NULL_DEREF,   0LL,           0LL,  64, 0 },
        // 13  1 << 40 on a 32-bit int              -> oversized shift, UB
        { SHIFT,        1LL,           40LL, 32, 0 },
        // 14  1 << 31 on a 32-bit int              -> defined (C++20)
        { SHIFT,        1LL,           31LL, 32, 0 },
        // 15  1 << -1                              -> negative shift count, UB
        { SHIFT,        1LL,           -1LL, 32, 0 },
        // 16  -1 << 1 on a 32-bit int              -> defined in C++20 (trap)
        { SHIFT,        -1LL,          1LL,  32, 0 },
        // 17  1LL << 63 on a 64-bit int            -> defined (C++20)
        { SHIFT,        1LL,           63LL, 64, 0 },
        // 18  read an int through a float* cast    -> strict-aliasing, UB
        { TYPE_PUN,     0LL,           0LL,  32, 0 },
        // 19  type-pun the same bytes via memcpy   -> defined
        { TYPE_PUN,     0LL,           0LL,  32, 1 },
        // 20  1LL << 64 on a 64-bit int            -> oversized shift, UB
        { SHIFT,        1LL,           64LL, 64, 0 },
    };

    long long packed = 0;   // bit vector packed MSB-first into an integer
    int count = 0;          // number of snippets flagged as UB
    for (int i = 0; i < 20; i++) {
        int bit = classify_ub(snips[i]) ? 1 : 0;
        printf("%d ", bit);
        packed = packed * 2 + bit;
        count += bit;
    }
    printf("\n");
    printf("packed=%lld\n", packed);
    printf("count=%d\n", count);
    return 0;
}
