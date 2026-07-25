// Reference: classic shared-memory tiled matmul, TILE=16 (compile-time
// constant, matching the fixed 256-thread block). One block computes
// one 16x16 output tile. The K dimension is swept in chunks of 16: each
// chunk, every thread cooperatively loads one element of A's tile and
// one element of B's tile into shared memory, __syncthreads() to make
// sure the WHOLE tile is loaded before anyone reads it, accumulates
// that chunk's 16 partial products, __syncthreads() again before the
// next chunk overwrites the shared tiles.
__global__ void tiled_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[256];
    __shared__ float Bs[256];

    int idx = threadIdx.x;
    int row = idx / 16;
    int col = idx % 16;
    float acc = 0.0;

    int kk = 0;
    while (kk < K) {
        As[row * 16 + col] = A[row * K + kk + col];
        Bs[row * 16 + col] = B[(kk + row) * N + col];
        __syncthreads();

        int k = 0;
        while (k < 16) {
            acc = acc + As[row * 16 + k] * Bs[k * 16 + col];
            k = k + 1;
        }
        __syncthreads();

        kk = kk + 16;
    }

    C[row * N + col] = acc;
}
