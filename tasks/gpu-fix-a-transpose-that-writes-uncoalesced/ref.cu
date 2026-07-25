// Fixed: stage the tile through shared memory. The load
// (tile[row*N+col] = in[row*N+col]) is already coalesced -- consecutive
// threads in a warp share `row` and read consecutive `col`, consecutive
// addresses. The fix is in the WRITE: instead of writing directly to
// the strided transposed address, write to out[row*N+col] (consecutive
// addresses again, since `row` is still fixed across the warp) and pull
// the already-transposed value out of shared memory instead:
// tile[col*N+row].
__global__ void transpose(const float* in, float* out, int N) {
    __shared__ float tile[1024];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N;
    int col = idx % N;

    tile[row * N + col] = in[row * N + col];
    __syncthreads();

    out[row * N + col] = tile[col * N + row];
}
