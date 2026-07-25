#include <algorithm>
#include "sol.hpp"

void tiled_matmul(int N, int T) {
    for (int ii = 0; ii < N; ii += T) {
        for (int jj = 0; jj < N; jj += T) {
            for (int kk = 0; kk < N; kk += T) {
                int i_end = std::min(ii + T, N);
                int j_end = std::min(jj + T, N);
                int k_end = std::min(kk + T, N);
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        for (int k = kk; k < k_end; k++) {
                            touch(a_addr(N, i, k));
                            touch(b_addr(N, k, j));
                            touch(c_addr(N, i, j));
                        }
                    }
                }
            }
        }
    }
}
