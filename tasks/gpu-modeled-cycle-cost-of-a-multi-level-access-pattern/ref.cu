// Reference: total modeled cycle cost of a fixed access trace. `level[i]`
// names which memory level access i went to (0=register, 1=shared,
// 2=L2, 3=DRAM/global); `latency[lvl]` is that level's per-access cycle
// cost. Single-threaded: sum latency[level[i]] over the whole trace.
__global__ void access_cost(float* out, const float* level, const float* latency, int n) {
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < n; i++) {
            int lvl = level[i];
            total = total + latency[lvl];
        }
        out[0] = total;
    }
}
