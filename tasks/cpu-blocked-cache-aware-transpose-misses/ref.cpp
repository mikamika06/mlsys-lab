#include "sol.hpp"

void blocked_transpose(int N, int B) {
    for (int ii = 0; ii < N; ii += B) {
        for (int jj = 0; jj < N; jj += B) {
            for (int i = ii; i < ii + B; i++) {
                for (int j = jj; j < jj + B; j++) {
                    touch(in_addr(N, i, j));
                    touch(out_addr(N, j, i));
                }
            }
        }
    }
}
