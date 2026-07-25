// Seeded dropout WITH inverted scaling, applied to real activations x.
// Hash h = (seed + i*2654435761) mod 1000000007, u = h/1000000007.0f, a
// pseudo-uniform value in [0, 1) determined by (seed, i) alone. If
// u < p: drop (out[i] = 0.0f). Otherwise keep, but divide by (1 - p) so
// the expected value stays x[i] regardless of p.
__global__ void triton_dropout(float* out, const float* x, int n, int seed, float p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // TODO: compute h, u, then out[i] = 0.0f if u < p else x[i] / (1 - p).
    }
}
