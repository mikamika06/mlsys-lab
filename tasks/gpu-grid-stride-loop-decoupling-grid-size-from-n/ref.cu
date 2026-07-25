// Reference: grid-stride loop. Each thread starts at its own global index
// and jumps forward by the TOTAL number of threads launched
// (blockDim.x * gridDim.x) each iteration, until it runs off the end of
// the array. This is correct for ANY grid size: a huge grid finishes every
// thread's loop after a single iteration (behaving like a normal one-
// element-per-thread launch), and a tiny grid just makes every thread loop
// more times to cover the same n -- the total amount of work covered
// never depends on how many threads happened to be launched.
__global__ void scale_grid_stride(float* out, const float* in, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i = i + stride) {
        out[i] = s * in[i];
    }
}
