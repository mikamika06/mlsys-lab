// TODO: TILE=16, block of 256 threads (row=idx/16, col=idx%16), one
// block computes the whole 16x16 output. Sweep kk from 0 to K in steps
// of 16: cooperatively load As[row*16+col]=A[row*K+kk+col] and
// Bs[row*16+col]=B[(kk+row)*N+col], __syncthreads(), accumulate
// sum_{k=0}^{15} As[row*16+k]*Bs[k*16+col] into acc, __syncthreads()
// again before the next chunk. Write C[row*N+col]=acc after the loop.
// See ref.cu.
__global__ void tiled_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[256];
    __shared__ float Bs[256];
    int idx = threadIdx.x;
    int row = idx / 16;
    int col = idx % 16;
    C[row * N + col] = 0.0;
}
