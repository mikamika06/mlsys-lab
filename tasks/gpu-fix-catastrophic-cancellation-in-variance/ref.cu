// Fixed: two-pass, mean-centered variance. Pass 1 computes the mean via
// a sequential-addressing tree reduction; pass 2 reduces the SQUARED
// DEVIATIONS FROM THAT MEAN (recomputed from the original x[], not from
// any single-pass moment) -- numerically stable regardless of how large
// the mean is relative to the true spread.
__global__ void row_variance(float* out, const float* x, int n) {
    __shared__ float sdata[32];
    int tid = threadIdx.x;
    sdata[tid] = x[tid];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2) {
        if (tid < stride) {
            sdata[tid] = sdata[tid] + sdata[tid + stride];
        }
        __syncthreads();
    }
    float mean = sdata[0] / n;
    __syncthreads();
    float dev = x[tid] - mean;
    sdata[tid] = dev * dev;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2) {
        if (tid < stride) {
            sdata[tid] = sdata[tid] + sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = sdata[0] / n;
    }
}
