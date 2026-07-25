// One warp (32 threads), one __shared__ buffer. `case_id` selects which
// of three named shared-memory access patterns every thread must
// realize (see task.md): 0 = conflict-free, 1 = broadcast, 2 = 4-way
// conflict. Each thread writes in[tid] to buf[idx] then reads it back
// into out[tid], where `idx` depends on case_id.
__global__ void bank_pattern(float* out, const float* in, int case_id, int n) {
    __shared__ float buf[128];
    int tid = threadIdx.x;
    // TODO: compute idx from case_id (0: idx=tid, 1: idx=0,
    // 2: idx=(tid%4)*32+(tid/4)), then buf[idx]=in[tid]; __syncthreads();
    // out[tid]=buf[idx];
}
