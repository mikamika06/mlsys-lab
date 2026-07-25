// Reference: a deterministic, per-element pseudo-random keep/drop mask.
// No RNG builtin exists in this CUDA-C subset (and no bitwise operators
// either, so no xorshift-style hash), so this models the standard
// "counter-based RNG" idea -- (seed, index) hashed independently, with no
// shared state and no sequential dependency between elements -- using
// pure multiplicative/modular arithmetic instead: thread i needs only
// `seed` and its OWN global index to reproduce the exact same decision
// every time, on any device, with any launch geometry.
__global__ void dropout_mask(float* mask, int n, int seed, float p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int h = (seed + i * 2654435761) % 1000000007;
        float u = h / 1000000007.0f;
        mask[i] = (u < p) ? 0.0f : 1.0f;
    }
}
