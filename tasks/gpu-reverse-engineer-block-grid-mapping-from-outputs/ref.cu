// Reference: test three known launch-encoding hypotheses against the
// observed array (blockDim is always 32; n is always 64) and report
// whichever one matches every element exactly.
//   flat:   out[t] = t                          (blockDim*gridDim == n, gridDim=2)
//   stride: out[t] = t % 32                      (grid-stride loop, gridDim=1)
//   2d:     out[t] = (t % 8) * 8 + (t / 8)        (transposed 8x8 index order, gridDim=2)
// result[0] = mapping_kind (0=flat, 1=stride, 2=2d); result[1] = gridDim.
__global__ void reconstruct_launch(const float* obs, int n, float* result) {
    int flat_mismatch = 0;
    int stride_mismatch = 0;
    int td_mismatch = 0;
    int t = 0;
    while (t < n) {
        float flat_val = t + 0.0;
        float stride_val = t % 32 + 0.0;
        float td_val = (t % 8) * 8 + (t / 8) + 0.0;
        if (obs[t] != flat_val) { flat_mismatch = flat_mismatch + 1; }
        if (obs[t] != stride_val) { stride_mismatch = stride_mismatch + 1; }
        if (obs[t] != td_val) { td_mismatch = td_mismatch + 1; }
        t = t + 1;
    }
    if (flat_mismatch == 0) {
        result[0] = 0.0;
        result[1] = 2.0;
    } else if (stride_mismatch == 0) {
        result[0] = 1.0;
        result[1] = 1.0;
    } else {
        result[0] = 2.0;
        result[1] = 2.0;
    }
}
