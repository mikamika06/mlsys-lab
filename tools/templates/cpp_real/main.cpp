#include <cstdio>
#include "sol.hpp"
int main() {
    const int N = 8;
    float x[N] = {-3.f, -1.f, 0.f, 1.f, 2.f, -5.f, 4.f, -0.5f};
    relu_inplace(x, N);
    double s = 0;
    for (int i = 0; i < N; i++) { printf("%.4f ", x[i]); s += x[i]; }
    printf("\nsum=%.4f\n", s);
    return 0;
}
