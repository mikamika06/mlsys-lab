// Reference: softmax over each 32-lane warp's row, entirely via warp
// shuffle -- no shared memory, no __syncthreads(). Both reductions use
// __shfl_xor_sync in a butterfly ladder (delta = 16,8,4,2,1): unlike
// __shfl_down_sync (which only leaves the total at lane 0), XOR-butterfly
// leaves the FINAL reduced value in EVERY lane, so no extra broadcast
// step is needed before computing each lane's own normalized output.
__global__ void warp_softmax(float* out, const float* x, int n) {
    int tid = threadIdx.x;
    float val = x[tid];

    float m = val;
    float t16 = __shfl_xor_sync(0xffffffff, m, 16); m = fmaxf(m, t16);
    float t8  = __shfl_xor_sync(0xffffffff, m, 8);  m = fmaxf(m, t8);
    float t4  = __shfl_xor_sync(0xffffffff, m, 4);  m = fmaxf(m, t4);
    float t2  = __shfl_xor_sync(0xffffffff, m, 2);  m = fmaxf(m, t2);
    float t1  = __shfl_xor_sync(0xffffffff, m, 1);  m = fmaxf(m, t1);

    float e = expf(val - m);
    float s = e;
    float u16 = __shfl_xor_sync(0xffffffff, s, 16); s = s + u16;
    float u8  = __shfl_xor_sync(0xffffffff, s, 8);  s = s + u8;
    float u4  = __shfl_xor_sync(0xffffffff, s, 4);  s = s + u4;
    float u2  = __shfl_xor_sync(0xffffffff, s, 2);  s = s + u2;
    float u1  = __shfl_xor_sync(0xffffffff, s, 1);  s = s + u1;

    out[tid] = e / s;
}
