// Per-row-quantized int8 matvec: y[i] = sum_j (w_int[i*N+j] * scale[i]) * x[j].
// One thread per output row (blockDim.x = M). Cache `x` in __shared__
// memory once instead of re-reading it from global per row, and
// dequantize each weight in a register right where it's used -- never
// write a dequantized copy of the weight matrix back to memory. See
// task.md.
__global__ void dequant_matvec(float* y, const float* w_int, const float* scale, const float* x, int M, int N) {
    // TODO
}
