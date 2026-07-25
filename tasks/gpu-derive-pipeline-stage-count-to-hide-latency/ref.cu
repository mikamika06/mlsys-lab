// Reference: minimal software-pipeline stage count needed to fully hide a
// per-tile load latency L behind C cycles of per-tile compute.
// stages = ceil(L / C) + 1.
__global__ void pipeline_stages(float* out, const float* L, const float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = ceilf(L[i] / C[i]) + 1.0f;
    }
}
