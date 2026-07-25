// TODO: for each t in [0, n), compute what the observed value WOULD be
// under each of the three hypotheses (flat: t; stride: t % 32; 2d:
// (t%8)*8 + t/8), count mismatches against obs[t] for each, and report
// whichever hypothesis has zero mismatches: result[0] = mapping_kind
// (0=flat, 1=stride, 2=2d), result[1] = that hypothesis's gridDim
// (2, 1, 2 respectively). See ref.cu.
__global__ void reconstruct_launch(const float* obs, int n, float* result) {
    result[0] = -1.0;
    result[1] = -1.0;
}
