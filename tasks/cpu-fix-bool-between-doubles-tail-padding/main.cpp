#include <cstdio>
#include "sol.hpp"

// FIXED driver. NAIVE_SIZE is sizeof(BadRecord) from the broken layout
// documented in sol.hpp -- fixed and independent of any candidate's answer.
constexpr double NAIVE_SIZE = 32.0;

int main() {
    size_t rsz = record_size();
    double ratio = static_cast<double>(rsz) / NAIVE_SIZE;

    printf("record_size=%zu\n", rsz);
    printf("offset_a=%zu\n", offset_a());
    printf("offset_b=%zu\n", offset_b());
    printf("offset_flag1=%zu\n", offset_flag1());
    printf("offset_flag2=%zu\n", offset_flag2());
    printf("size_ratio=%.6f\n", ratio);
    return 0;
}
