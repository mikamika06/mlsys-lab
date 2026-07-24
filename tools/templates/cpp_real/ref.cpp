#include "sol.hpp"
void relu_inplace(float* x, int n) {
    for (int i = 0; i < n; i++) if (x[i] < 0.f) x[i] = 0.f;
}
