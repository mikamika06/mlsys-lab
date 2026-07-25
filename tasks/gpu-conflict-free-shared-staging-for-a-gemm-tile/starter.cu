// Single 16x16x16 GEMM tile, staged through shared memory. thread tid maps
// to (col = tid/16, row = tid%16). Stage A and B into __shared__ arrays
// As/Bs (choose your own row stride -- it does not have to equal n), then
// reduce C[row][col] = sum_k A[row][k] * B[k][col] by reading back out of
// shared memory, not global memory, inside the k loop.
__global__ void gemm_tile(float* C, const float* A, const float* B, int n) {
    __shared__ float As[256];
    __shared__ float Bs[256];
    int tid = threadIdx.x;
    int col = tid / n;
    int row = tid % n;

    As[row * n + col] = A[row * n + col];
    Bs[row * n + col] = B[row * n + col];
    __syncthreads();

    float acc = 0.0f;
    for (int k = 0; k < n; k = k + 1) {
        acc = acc + As[row * n + k] * Bs[k * n + col];
    }
    C[row * n + col] = acc;
}
