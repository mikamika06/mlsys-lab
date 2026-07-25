// Reference: one warp (32 threads), one __shared__ buffer. `case_id`
// selects which of three named shared-memory access patterns every
// thread realizes, writing then reading back through `buf[idx]`:
//
//   case 0: conflict-free -- idx = tid, each thread hits a distinct word
//           in a distinct bank (bank = idx % 32).
//   case 1: broadcast     -- idx = 0 for every thread, all 32 threads
//           hit the exact SAME word (a real broadcast, not a conflict).
//   case 2: 4-way conflict -- idx = (tid % 4) * 32 + (tid / 4); the 4
//           threads sharing a bank (tid/4 fixed) each hit a DIFFERENT
//           word (bank = idx % 32 = tid / 4), a genuine conflict.
__global__ void bank_pattern(float* out, const float* in, int case_id, int n) {
    __shared__ float buf[128];
    int tid = threadIdx.x;
    int idx;
    if (case_id == 0) {
        idx = tid;
    } else if (case_id == 1) {
        idx = 0;
    } else {
        idx = (tid % 4) * 32 + (tid / 4);
    }
    buf[idx] = in[tid];
    __syncthreads();
    out[tid] = buf[idx];
}
