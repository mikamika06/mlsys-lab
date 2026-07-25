// BUG: the hash only depends on `seed`, never on `i`. Every thread
// computes the exact same value of `h`, so `out[i]` is the exact same
// coin flip broadcast to EVERY element -- the mask ends up either all
// 1.0 or all 0.0 instead of an independent draw per element. Fix it by
// mixing the per-element index `i` into the hash input so each thread
// samples a different point in the hash's output space.
__global__ void dropout_mask(int seed, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int h = seed * 2654435761;
        h = h % 2147483647;
        float r = h / 2147483647.0;
        out[i] = r >= 0.5 ? 1.0 : 0.0;
    }
}
