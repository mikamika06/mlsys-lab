// Implement both inclusive-scan algorithms over n=256 elements, single
// block -- see the reference description in the task.

__global__ void hillis_steele_scan(float* out, const float* in, int n) {
    __shared__ float temp[512];
    int tid = threadIdx.x;
    temp[tid] = in[tid];
    __syncthreads();
    // your code here
}

__global__ void blelloch_scan(float* out, const float* in, int n) {
    __shared__ float temp[256];
    int tid = threadIdx.x;
    temp[tid] = in[tid];
    __syncthreads();
    // your code here
}
