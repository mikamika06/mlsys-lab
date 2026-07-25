// Reference: RMSNorm forward over one 32-element row (one block).
// A sequential-addressing tree reduction sums x[tid]^2 into sdata[0];
// every thread reads that same value to compute the shared RMS, then
// scales its own element: out[tid] = (x[tid] / rms) * gamma[tid].
__global__ void rmsnorm_forward(float* out, const float* x, const float* gamma, float eps, int n) {
    __shared__ float sdata[32];
    int tid = threadIdx.x;
    float v = x[tid];
    sdata[tid] = v * v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2) {
        if (tid < stride) {
            sdata[tid] = sdata[tid] + sdata[tid + stride];
        }
        __syncthreads();
    }
    float ms = sdata[0] / n;
    float rms = sqrtf(ms + eps);
    out[tid] = (x[tid] / rms) * gamma[tid];
}
