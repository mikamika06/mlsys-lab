// Total modeled cycle cost of a fixed access trace: sum latency[level[i]]
// over i in [0,n), single-threaded, where level[i] in {0,1,2,3} names
// the memory level (register/shared/L2/DRAM) access i went to.
__global__ void access_cost(float* out, const float* level, const float* latency, int n) {
    // TODO: if (threadIdx.x == 0), accumulate latency[(int)level[i]] for
    // i = 0..n-1, then out[0] = total.
}
