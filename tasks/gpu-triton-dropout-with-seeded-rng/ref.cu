// Reference: seeded dropout WITH inverted scaling, applied to real
// activations (not just a keep/drop mask). Same counter-based hash as a
// bare dropout mask -- (seed, index) alone determines the decision, no
// RNG builtin, no bitwise ops available in this CUDA-C subset, so the
// hash is pure multiplication/modulo. A surviving element is divided by
// (1 - p) ("inverted scaling"): that keeps E[out[i]] == x[i] regardless of
// p, so downstream code doesn't need to know dropout even ran.
__global__ void triton_dropout(float* out, const float* x, int n, int seed, float p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int h = (seed + i * 2654435761) % 1000000007;
        float u = h / 1000000007.0f;
        if (u < p) {
            out[i] = 0.0f;
        } else {
            out[i] = x[i] / (1.0f - p);
        }
    }
}
