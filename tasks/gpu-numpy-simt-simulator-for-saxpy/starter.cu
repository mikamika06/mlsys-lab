// TODO: implement SAXPY. Each thread i (i = blockIdx.x*blockDim.x + threadIdx.x)
// should compute y[i] = a*x[i] + y[i], guarded by i < n.
__global__ void saxpy_kernel(float* y, const float* x, int n, float a) {
}
