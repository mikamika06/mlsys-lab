#include "sol.hpp"

void blocked_transpose(const double* in, double* out, int n, int block) {
    for (int bi = 0; bi < n; bi += block) {
        for (int bj = 0; bj < n; bj += block) {
            for (int i = bi; i < bi + block; ++i) {
                for (int j = bj; j < bj + block; ++j) {
                    out[j * n + i] = in[i * n + j];
                }
            }
        }
    }
}
