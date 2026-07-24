#include "sol.hpp"

// TODO: implement stridedRowSums per sol.hpp's contract.
void stridedRowSums(const uint8_t* buf, int M, int N,
                     long strideRow, long strideCol, long fieldOffset,
                     double* out) {
    // your code here
    for (int i = 0; i < M; i++) out[i] = 0.0;
}
