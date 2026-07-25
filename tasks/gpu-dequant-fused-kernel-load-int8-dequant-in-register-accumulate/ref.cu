// Reference: per-row-quantized int8 matvec, dequant fused into the
// accumulate loop. One thread per output row i (blockDim.x = M).
//
// Activations are cooperatively loaded into __shared__ memory ONCE
// (thread i loads x[i] into xs[i]) and reused by every row's reduction,
// instead of every one of the M threads re-reading the same N values
// straight from global memory.
//
// Each weight `w_int[i*N+j]` is read from global memory EXACTLY ONCE,
// dequantized immediately in a register (`w_int[...] * s`, where `s`
// is this row's own quantization scale, loaded once per thread), and
// consumed the same instruction it's produced in -- no dequantized
// copy of the weight matrix is ever written back to memory.
__global__ void dequant_matvec(float* y, const float* w_int, const float* scale, const float* x, int M, int N) {
    int i = threadIdx.x;

    __shared__ float xs[8];
    xs[i] = x[i];
    __syncthreads();

    float s = scale[i];
    float acc = 0.0f;
    for (int j = 0; j < N; j++) {
        float w_fp = w_int[i * N + j] * s;
        acc = acc + w_fp * xs[j];
    }
    y[i] = acc;
}
