// stages = ceil(L[i] / C[i]) + 1 -- the fewest pipeline stages that fully
// hide a per-tile load latency L behind C cycles of per-tile compute.
__global__ void pipeline_stages(float* out, const float* L, const float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard i < n, then out[i] = ceilf(L[i] / C[i]) + 1.0f;
}
