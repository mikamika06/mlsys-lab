// One attention tile step: out[i][j] = sum_d Q[i][d] * K[j][d] for the
// BQ x BK score tile. Load Q_tile and K_tile into __shared__ memory
// ONCE (cooperatively, one element per thread), then have every thread
// compute its own dot product out of shared memory + a register
// accumulator -- never re-reading Q or K from global memory inside the
// reduction loop. See task.md.
__global__ void qk_tile(float* out, const float* Q, const float* K, int BQ, int BK, int D) {
    // TODO
}
