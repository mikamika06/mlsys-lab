#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include "sol.hpp"

struct ElemA { char c; double val; };
struct ElemB { int a; float b; double val; };

static void writeElemA(uint8_t* p, ElemA e) { std::memcpy(p, &e, sizeof(ElemA)); }
static void writeElemB(uint8_t* p, ElemB e) { std::memcpy(p, &e, sizeof(ElemB)); }

static void printRow(const char* tag, const double* out, int M) {
    printf("%s", tag);
    for (int i = 0; i < M; i++) printf(" %.6f", out[i]);
    printf("\n");
}

int main() {
    // Scenario 1: parent grid 8x8 of ElemA, row-major contiguous; read a
    // strided sub-slice with shape 4x4 and step 2 in both dimensions.
    {
        const int Mp = 8, Np = 8;
        long rowStride = (long)Np * sizeof(ElemA);
        long colStride = (long)sizeof(ElemA);
        std::vector<uint8_t> buf((size_t)(Mp * rowStride));
        for (int i = 0; i < Mp; i++)
            for (int j = 0; j < Np; j++) {
                ElemA e{(char)('a' + (i + j) % 26), (double)(i * 10 + j + 1)};
                writeElemA(buf.data() + i * rowStride + j * colStride, e);
            }

        int M = 4, N = 4;
        long subRowStride = 2 * rowStride;
        long subColStride = 2 * colStride;
        long fieldOff = (long)offsetof(ElemA, val);
        std::vector<double> out(M);
        stridedRowSums(buf.data(), M, N, subRowStride, subColStride, fieldOff, out.data());
        printRow("scenario1", out.data(), M);
    }

    // Scenario 2: parent grid 6x10 of ElemB, target the 3rd field (val);
    // read a strided sub-slice with shape 3x5 and step 2 in both dims.
    {
        const int Mp = 6, Np = 10;
        long rowStride = (long)Np * sizeof(ElemB);
        long colStride = (long)sizeof(ElemB);
        std::vector<uint8_t> buf((size_t)(Mp * rowStride));
        for (int i = 0; i < Mp; i++)
            for (int j = 0; j < Np; j++) {
                ElemB e{i, (float)j, (double)(i * 5 + j) + 0.5};
                writeElemB(buf.data() + i * rowStride + j * colStride, e);
            }

        int M = 3, N = 5;
        long subRowStride = 2 * rowStride;
        long subColStride = 2 * colStride;
        long fieldOff = (long)offsetof(ElemB, val);
        std::vector<double> out(M);
        stridedRowSums(buf.data(), M, N, subRowStride, subColStride, fieldOff, out.data());
        printRow("scenario2", out.data(), M);
    }

    // Scenario 3: transposed layout -- stored "column-major" (consecutive
    // rows are adjacent, columns are M elements apart).
    {
        const int M = 4, N = 3;
        long strideRow = (long)sizeof(ElemA);
        long strideCol = (long)M * sizeof(ElemA);
        std::vector<uint8_t> buf((size_t)(M * N) * sizeof(ElemA));
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                ElemA e{'x', (double)(i * 5 + j + 1)};
                writeElemA(buf.data() + i * strideRow + j * strideCol, e);
            }

        long fieldOff = (long)offsetof(ElemA, val);
        std::vector<double> out(M);
        stridedRowSums(buf.data(), M, N, strideRow, strideCol, fieldOff, out.data());
        printRow("scenario3", out.data(), M);
    }

    return 0;
}
