// RMSNorm forward over one 32-element row (one block of 32 threads):
// out[tid] = (x[tid] / sqrt(mean(x^2) + eps)) * gamma[tid]. Compute
// mean(x^2) with a sequential-addressing tree reduction in shared memory
// (every thread needs the SAME final sum, so read it after the barrier).
__global__ void rmsnorm_forward(float* out, const float* x, const float* gamma, float eps, int n) {
    __shared__ float sdata[32];
    int tid = threadIdx.x;
    // TODO: sdata[tid] = x[tid]*x[tid]; barrier; sequential-addressing
    // sum reduction (barrier each step); ms = sdata[0]/n; rms =
    // sqrtf(ms+eps); out[tid] = (x[tid]/rms) * gamma[tid].
}
