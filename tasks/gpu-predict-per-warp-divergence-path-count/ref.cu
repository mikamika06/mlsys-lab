// Reference: evaluate the data-dependent predicate x[i] > threshold
// for every element. The grader aggregates these per-thread results
// into per-warp path counts itself (1 if a warp's 32 lanes all agree,
// 2 if they split) -- this kernel's only job is to get the per-element
// predicate right.
__global__ void eval_predicate(const float* x, float* pred_out, int n, float threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pred_out[i] = x[i] > threshold ? 1.0 : 0.0;
    }
}
