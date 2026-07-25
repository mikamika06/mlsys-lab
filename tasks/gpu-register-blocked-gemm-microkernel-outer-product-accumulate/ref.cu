// Register-blocked GEMM microkernel: instead of one thread per output
// element, each thread owns a 2x2 TILE of C, held entirely in its own
// registers (acc00..acc11 -- plain local scalars, never spilled to
// shared or global memory). At every step of the K loop, the thread
// loads only 2 elements of A (one per output ROW it owns) and 2 of B
// (one per output COLUMN it owns) -- 4 loads -- and forms the OUTER
// PRODUCT of those two length-2 vectors: 4 FMAs from 4 loads, instead
// of the 8 loads four separate scalar-output threads would need for
// the same 4 FMAs. Thread coarsening amortizes memory traffic over more
// arithmetic.
__global__ void gemm_regblock(float* C, const float* A, const float* B, int M, int N, int K) {
    int tid = threadIdx.x;
    int tilesPerRow = N / 2;
    int tileCol = tid % tilesPerRow;
    int tileRow = tid / tilesPerRow;
    int row0 = tileRow * 2;
    int col0 = tileCol * 2;

    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    int k = 0;
    while (k < K) {
        float a0 = A[row0 * K + k];
        float a1 = A[(row0 + 1) * K + k];
        float b0 = B[k * N + col0];
        float b1 = B[k * N + col0 + 1];
        acc00 += a0 * b0;
        acc01 += a0 * b1;
        acc10 += a1 * b0;
        acc11 += a1 * b1;
        k = k + 1;
    }

    C[row0 * N + col0] = acc00;
    C[row0 * N + col0 + 1] = acc01;
    C[(row0 + 1) * N + col0] = acc10;
    C[(row0 + 1) * N + col0 + 1] = acc11;
}
