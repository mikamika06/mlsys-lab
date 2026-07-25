// Fixed: each thread's hash input is (seed + i), not just `seed` --
// every element gets its own independent point in the hash's output
// space, instead of every thread computing the exact same hash and
// broadcasting one shared coin flip to the whole array.
__global__ void dropout_mask(int seed, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int h = (seed + i) * 2654435761;
        h = h % 2147483647;
        float r = h / 2147483647.0;
        out[i] = r >= 0.5 ? 1.0 : 0.0;
    }
}
