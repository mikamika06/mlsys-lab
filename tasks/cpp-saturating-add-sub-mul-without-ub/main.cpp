#include "sol.hpp"
#include <cstdio>

int main() {
    static const int32_t a[] = {2147483647, -2147483648, 100000, 2147483647,
                                 -100, -1073741824, 0, 50000};
    static const int32_t b[] = {100, 100, 200000, 2,
                                 2147483647, 3, -2147483648, 50000};
    const int n = 8;

    static const Op ops[] = {Op::Add, Op::Sub, Op::Mul};
    int32_t out[n];
    for (Op op : ops) {
        saturating_arithmetic(a, b, n, op, out);
        for (int i = 0; i < n; i++) {
            printf("%d\n", out[i]);
        }
    }
    return 0;
}
