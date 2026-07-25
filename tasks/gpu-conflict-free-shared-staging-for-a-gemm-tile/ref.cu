// Reference: single 16x16x16 GEMM tile, staged through padded shared
// memory. thread tid maps to (col = tid/16, row = tid%16), so a 32-lane
// warp spans 2 full COLUMNS (all 16 rows of each). Both As and Bs are
// stored with a padded row stride (n+1, not n): with the row-varies-fastest
// access pattern this kernel uses on As[row*lda+k] during the reduction,
// an unpadded (stride n) layout would send every 2 threads 16 apart in
// `row` to the same shared-memory bank; padding breaks that alignment
// (gcd(n+1, 32) == 1) so every lane lands on a distinct bank instead.
__global__ void gemm_tile(float* C, const float* A, const float* B, int n) {
    __shared__ float As[272];
    __shared__ float Bs[272];
    int tid = threadIdx.x;
    int col = tid / n;
    int row = tid % n;
    int lda = n + 1;
    int ldb = n + 1;

    As[row * lda + col] = A[row * n + col];
    Bs[row * ldb + col] = B[row * n + col];
    __syncthreads();

    float acc = 0.0f;
    for (int k = 0; k < n; k = k + 1) {
        acc = acc + As[row * lda + k] * Bs[k * ldb + col];
    }
    C[row * n + col] = acc;
}
