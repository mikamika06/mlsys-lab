#include "sol.hpp"
#include <cstring>

void stridedRowSums(const uint8_t* buf, int M, int N,
                     long strideRow, long strideCol, long fieldOffset,
                     double* out) {
    for (int i = 0; i < M; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            const uint8_t* p = buf + (long)i * strideRow + (long)j * strideCol + fieldOffset;
            double v;
            std::memcpy(&v, p, sizeof(double));
            sum += v;
        }
        out[i] = sum;
    }
}
