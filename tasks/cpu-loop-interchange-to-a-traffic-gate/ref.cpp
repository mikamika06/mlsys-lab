#include "sol.hpp"

void row_major_traverse(int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            touch(elem_addr(N, row, col));
        }
    }
}
