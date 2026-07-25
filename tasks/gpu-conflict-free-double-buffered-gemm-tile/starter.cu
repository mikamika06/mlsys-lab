// Compute C = A*B for 32x32 x 32x32 matrices (K=64, two 32-wide K-tiles),
// one thread per output element, launched as a single block of 1024
// (=32*32) threads: col = threadIdx.x / 32, row = threadIdx.x % 32.
//
// Use TWO padded (33-word stride) shared-memory buffers per operand so
// the next K-tile can be prefetched into the buffer NOT currently being
// read from: preload K-tile 0 into buffer 0 before the loop; each loop
// iteration prefetches K-tile kt+1 into the other buffer (if any remain)
// before accumulating this iteration's 32-term dot product from the
// CURRENT buffer; out[row][col] = accumulated sum after both K-tiles.
// See task.md for the exact indexing.
__global__ void gemm_tile_dbuf(float* C, const float* A, const float* B, int M, int N, int K) {
    __shared__ float As[2112];
    __shared__ float Bs[2112];
    int tid = threadIdx.x;
    int col = tid / 32;
    int row = tid % 32;
    // TODO: prologue load, syncthreads, then the double-buffered K-tile
    // loop (prefetch + accumulate + syncthreads), then C[row*N+col] = acc.
}
