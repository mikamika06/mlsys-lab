// TODO: pred_out[i] = 1.0 if x[i] > threshold, else 0.0. See ref.cu.
__global__ void eval_predicate(const float* x, float* pred_out, int n, float threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pred_out[i] = 0.0;
    }
}
