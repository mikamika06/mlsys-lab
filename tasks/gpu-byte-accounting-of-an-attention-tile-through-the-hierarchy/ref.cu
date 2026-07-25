// Reference: one attention tile step, Q_tile @ K_tile^T, with the
// global -> shared -> register hierarchy done right.
//
// COOPERATIVE LOAD (global -> shared, paid ONCE): with BQ*D == BK*D ==
// BQ*BK == 64 threads in this task's fixed sizing, thread `tid` loads
// exactly one Q element and one K element into shared memory --
// Q[0..63] and K[0..63] are each touched by global memory exactly once
// across the whole block, not once per consumer.
//
// COMPUTE (shared -> register, reused BQ*BK times): after the barrier,
// each thread (i, j) walks its own dot product entirely out of the
// on-chip tile (Qs, Ks) and a running register accumulator `acc`,
// touching global memory zero more times. Every one of the D=8
// multiply-adds that make up one output score is a shared-memory read
// plus a register update, never a global load.
__global__ void qk_tile(float* out, const float* Q, const float* K, int BQ, int BK, int D) {
    int tid = threadIdx.x;

    __shared__ float Qs[64];
    __shared__ float Ks[64];
    Qs[tid] = Q[tid];
    Ks[tid] = K[tid];
    __syncthreads();

    int i = tid / BK;
    int j = tid % BK;
    float acc = 0.0f;
    for (int d = 0; d < D; d++) {
        acc = acc + Qs[i * D + d] * Ks[j * D + d];
    }
    out[i * BK + j] = acc;
}
