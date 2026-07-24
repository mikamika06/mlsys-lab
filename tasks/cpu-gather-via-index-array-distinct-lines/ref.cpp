#include "sol.hpp"

void gather(const float* base, const int* idx, int n, float* result) {
    for (int i = 0; i < n; ++i) {
        touch(&base[idx[i]]);
        result[i] = base[idx[i]];
    }
}
