// Reference: a from-scratch counter-based RNG. Thread i's output depends
// ONLY on its own counter[i] and the shared key -- no thread reads any
// other thread's state, and there is no serial carry between outputs, so
// every lane can generate its value fully in parallel.
//
// 3 mixing rounds of a multiply-add-mod hash, each round using a
// different additive constant (round index r) so the rounds aren't just
// repeats of each other:
//   x = (x * 48271 + key + r * 7919) mod 1000003
// Final output is x normalized into [0, 1).
__global__ void philox_style_rng(float* out, const float* counters, float key, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = counters[i];
        for (int r = 0; r < 3; r++) {
            x = (x * 48271.0f + key + r * 7919.0f) % 1000003.0f;
        }
        out[i] = x / 1000003.0f;
    }
}
