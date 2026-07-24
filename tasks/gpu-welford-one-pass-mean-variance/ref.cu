// Reference: single-thread (lane 0 only) Welford one-pass mean/variance.
__global__ void welford_kernel(const float* x, float* out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid != 0) return;

    int count = 0;
    float mean = 0.0;
    float M2 = 0.0;

    for (int i = 0; i < n; i++) {
        float xi = x[i];
        count = count + 1;
        float delta = xi - mean;
        mean = mean + delta / count;
        float delta2 = xi - mean;
        M2 = M2 + delta * delta2;
    }

    float var = M2 / n;
    out[0] = mean;
    out[1] = var;
}
