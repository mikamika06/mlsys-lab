#include <cstdio>
#include "sol.hpp"

int main() {
    // Deterministic integer fixture: a[i] = i^3 - 7i + 3, for i in [0, 64).
    const int L = 64;
    long long a[L];
    for (int i = 0; i < L; i++)
        a[i] = (long long)i * i * i - 7LL * i + 3LL;

    // (n, stride) pairs chosen so the last index (n-1)*stride stays in [0, 64).
    const int NF = 5;
    int ns[NF]      = {64, 32, 22, 13, 10};
    int strides[NF] = { 1,  2,  3,  5,  7};

    long long total = 0;
    for (int f = 0; f < NF; f++) {
        long long s = strided_weighted_sum(a, ns[f], strides[f]);
        printf("%lld ", s);
        total += s;
    }
    printf("\ntotal=%lld\n", total);
    return 0;
}
