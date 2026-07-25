// BUG: correct, but writes uncoalesced. Consecutive threads in a warp
// share `row` and have consecutive `col` -- so the READ (in[row*N+col])
// is coalesced -- but the WRITE address (out[col*N+row]) has `col`
// varying fastest as the OUTER index, so consecutive threads' writes
// land N floats apart: every write in the warp hits a different
// 128-byte segment. Fix it by staging the tile through shared memory
// (see task.md) so the write becomes contiguous too.
__global__ void transpose(const float* in, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N;
    int col = idx % N;
    out[col * N + row] = in[row * N + col];
}
