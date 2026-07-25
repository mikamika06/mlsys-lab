// Reference: naive transpose. Thread tid handles input element tid
// (coalesced read: consecutive threads read consecutive addresses of
// `in`), and writes it to its transposed position in `out` (scattered:
// consecutive threads' output addresses are `rows` apart, since out is
// stored cols x rows row-major).
__global__ void naive_transpose(float* out, const float* in, int rows, int cols) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (tid < n) {
        int r = tid / cols;
        int c = tid % cols;
        out[c * rows + r] = in[tid];
    }
}
