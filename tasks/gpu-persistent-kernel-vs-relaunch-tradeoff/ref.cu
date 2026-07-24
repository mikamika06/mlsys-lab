// Strategy 1: launch ONCE, each thread loops K times internally.
__global__ void persistent_kernel(float* gmem, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = gmem[i];
        for (int k = 0; k < K; k++) {
            v = v + 1.0;
        }
        gmem[i] = v;
    }
}

// Strategy 2: one iteration's worth of work; the host launches this K times.
__global__ void relaunch_kernel(float* gmem, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        gmem[i] = gmem[i] + 1.0;
    }
}

// Host-side overhead model, computed on the GPU by a single thread so the
// whole task stays real compiled/executed CUDA-C:
//   persistent = H + K * C_iter
//   relaunch   = K * (H + C_iter)
__global__ void model_launch_cycles_kernel(float* out, int launch_overhead, int compute_cost_per_iter, int K) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = launch_overhead + K * compute_cost_per_iter;
        out[1] = K * (launch_overhead + compute_cost_per_iter);
    }
}
