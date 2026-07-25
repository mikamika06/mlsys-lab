// Fixed: `bias[0]` is the exact same broadcast address on every single
// iteration -- every thread in every warp always wants that one value.
// Load it ONCE, before the loop, into a local variable, and reuse it --
// instead of re-issuing a fresh global access step for it on every one
// of the K iterations.
__global__ void biased_sum(const float* x, const float* bias, float* out, int n, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float b = bias[0];
        float acc = 0.0;
        int k = 0;
        while (k < K) {
            acc = acc + x[k * n + i] + b;
            k = k + 1;
        }
        out[i] = acc;
    }
}
