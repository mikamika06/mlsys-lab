// "One program per row" softmax: block `blockIdx.x` owns row `blockIdx.x`
// end to end. For each row: find its max, sum exp(x - max) across the
// row, then write exp(x - max) / sum for every element -- max-shifting
// keeps every exponent <= 0 so nothing overflows, no matter how large the
// row's raw values are.
__global__ void row_softmax(float* out, const float* x, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        int base = row * cols;
        // TODO: m = max over x[base .. base+cols-1];
        //       sum = sum of expf(x[base+j] - m) over the row;
        //       out[base+j] = expf(x[base+j] - m) / sum for every j.
    }
}
