// Reference: thread coarsening with C=2 -- each thread owns a 2x2
// output micro-tile instead of a single element. At every k, it loads
// exactly 2 values from A (one per output row it owns) and 2 from B
// (one per output column it owns) into local variables (registers),
// then reuses each loaded value across BOTH of the outputs that need
// it: a0 feeds both acc00 and acc01, b0 feeds both acc00 and acc10 --
// 4 multiply-adds from 4 loads, instead of 4 independent single-output
// threads needing 4+4=8 loads for the same 4 results.
__global__ void coarsened_matmul(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tiles_per_row = N / 2;
    if (idx < tiles_per_row * tiles_per_row) {
        int ti = idx / tiles_per_row;
        int tj = idx % tiles_per_row;
        int row0 = ti * 2;
        int row1 = row0 + 1;
        int col0 = tj * 2;
        int col1 = col0 + 1;

        float acc00 = 0.0;
        float acc01 = 0.0;
        float acc10 = 0.0;
        float acc11 = 0.0;

        int k = 0;
        while (k < N) {
            float a0 = A[row0 * N + k];
            float a1 = A[row1 * N + k];
            float b0 = B[k * N + col0];
            float b1 = B[k * N + col1];

            acc00 = acc00 + a0 * b0;
            acc01 = acc01 + a0 * b1;
            acc10 = acc10 + a1 * b0;
            acc11 = acc11 + a1 * b1;

            k = k + 1;
        }

        C[row0 * N + col0] = acc00;
        C[row0 * N + col1] = acc01;
        C[row1 * N + col0] = acc10;
        C[row1 * N + col1] = acc11;
    }
}
