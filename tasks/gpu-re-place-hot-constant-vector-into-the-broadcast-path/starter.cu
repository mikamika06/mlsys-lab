// BUG: `bias[0]` never changes -- every thread, every iteration, wants
// the exact same address -- but this kernel re-reads it from global
// memory on every one of the K loop iterations instead of loading it
// once. The result is correct, but every one of those K-1 redundant
// re-reads is a real extra global-memory access step. Fix it by
// loading `bias[0]` into a local variable ONCE, before the loop, and
// reusing that instead of indexing `bias` again inside the loop.
__global__ void biased_sum(const float* x, const float* bias, float* out, int n, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float acc = 0.0;
        int k = 0;
        while (k < K) {
            acc = acc + x[k * n + i] + bias[0];
            k = k + 1;
        }
        out[i] = acc;
    }
}
