#include <cstdio>
#include "sol.hpp"

// Fixed driver: 20 deterministic (bit pattern, width, signedness) cases.
// Prints the decoded decimal value of each, then their sum.
int main() {
    struct Case { unsigned long long bits; int width; int is_signed; };
    const Case cases[] = {
        {0xFFull,       8,  1},   // -1
        {0xFFull,       8,  0},   // 255
        {0x80ull,       8,  1},   // -128
        {0x7Full,       8,  1},   // 127
        {0x00ull,       8,  0},   // 0
        {0xFFFFull,     16, 1},   // -1
        {0xFFFFull,     16, 0},   // 65535
        {0x8000ull,     16, 1},   // -32768
        {0x1234ull,     16, 1},   // 4660
        {0xABCDull,     16, 1},   // -21555
        {0xFFFFFFFFull, 32, 1},   // -1
        {0xFFFFFFFFull, 32, 0},   // 4294967295
        {0x80000000ull, 32, 1},   // -2147483648
        {0x80000000ull, 32, 0},   // 2147483648
        {0xDEADBEEFull, 32, 1},   // -559038737
        {0x0000CAFEull, 32, 1},   // 51966
        {0xFull,        4,  1},   // -1
        {0x8ull,        4,  1},   // -8
        {0x7ull,        4,  1},   // 7
        {0x123456ull,   24, 1},   // 1193046
    };
    const int N = (int)(sizeof(cases) / sizeof(cases[0]));
    long long sum = 0;
    for (int i = 0; i < N; i++) {
        long long v = twos_complement_value(cases[i].bits, cases[i].width, cases[i].is_signed);
        printf("%lld ", v);
        sum += v;
    }
    printf("\nsum=%lld\n", sum);
    return 0;
}
