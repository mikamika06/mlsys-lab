// A from-scratch counter-based RNG: thread i's output must depend ONLY on
// its own counters[i] and the shared key (no cross-thread dependency).
// 3 rounds of x = (x*48271 + key + r*7919) mod 1000003, starting from
// x = counters[i], then out[i] = x / 1000003.
__global__ void philox_style_rng(float* out, const float* counters, float key, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard i < n, run the 3-round hash starting from x =
    // counters[i], and write out[i] = x / 1000003.
}
