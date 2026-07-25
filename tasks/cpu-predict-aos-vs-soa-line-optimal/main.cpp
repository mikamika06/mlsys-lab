#include <cstdio>
#include "sol.hpp"

int main() {
    // P1: 1000 records of 4 x 4-byte fields, read 1 of 4 fields.
    const int fb1[4] = {4, 4, 4, 4};
    const bool m1[4] = {true, false, false, false};
    int l1 = soa_is_optimal(1000, 4, fb1, m1);

    // P2: same records, read 2 of 4 fields.
    const bool m2[4] = {true, true, false, false};
    int l2 = soa_is_optimal(1000, 4, fb1, m2);

    // P3: same records, read 3 of 4 fields.
    const bool m3[4] = {true, true, true, false};
    int l3 = soa_is_optimal(1000, 4, fb1, m3);

    // P4: same records, read ALL 4 fields (a full-record traversal).
    const bool m4[4] = {true, true, true, true};
    int l4 = soa_is_optimal(1000, 4, fb1, m4);

    // P5: 500 records of 5 x 4-byte fields, read 2 non-adjacent fields.
    const int fb5[5] = {4, 4, 4, 4, 4};
    const bool m5[5] = {true, false, true, false, false};
    int l5 = soa_is_optimal(500, 5, fb5, m5);

    printf("labels=%d,%d,%d,%d,%d\n", l1, l2, l3, l4, l5);
    return 0;
}
