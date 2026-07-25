// Reference: "one program per row" softmax, Triton-style -- each block
// (blockIdx.x = row index) owns exactly one row end to end: find its max,
// sum its shifted exponentials, then normalize. Single thread per block
// is enough to express the row-owned loop in this CUDA-C subset; the
// important property is that row `r`'s three passes never touch any
// other row's data.
__global__ void row_softmax(float* out, const float* x, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        int base = row * cols;

        float m = x[base + 0];
        for (int j = 1; j < cols; j = j + 1) {
            m = fmaxf(m, x[base + j]);
        }

        float sum = 0.0f;
        for (int j = 0; j < cols; j = j + 1) {
            sum = sum + expf(x[base + j] - m);
        }

        for (int j = 0; j < cols; j = j + 1) {
            out[base + j] = expf(x[base + j] - m) / sum;
        }
    }
}
