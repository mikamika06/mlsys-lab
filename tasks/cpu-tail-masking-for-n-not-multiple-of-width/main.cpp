#include <cstdio>
#include "sol.hpp"

// FIXED driver. N = 22 is deliberately NOT a multiple of WIDTH = 4
// (22 % 4 == 2), so 2 trailing elements form the tail.
constexpr int N = 22;

int main() {
    static float a[N], b[N], c[N];
    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i) * 0.5f;
        c[i] = -999.0f; // sentinel: never a valid a[i]+b[i] result here
    }

    vec_add(a, b, c, N);

    for (int i = 0; i < N; ++i) {
        printf("c[%d]=%.6f\n", i, c[i]);
    }
    return 0;
}
