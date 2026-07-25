// TODO: concurrency = warps_resident * ilp. exposed = max(0, mem_latency
// - concurrency) -- the portion of the memory stall that concurrency
// couldn't cover. out[0] = compute_cycles + exposed. See ref.cu (the
// "+ 0.0" there forces float arithmetic for the subtraction/comparison).
__global__ void latency_hiding_cycles(int warps_resident, int ilp, int compute_cycles,
                                       int mem_latency, float* out) {
    out[0] = 0.0;
}
