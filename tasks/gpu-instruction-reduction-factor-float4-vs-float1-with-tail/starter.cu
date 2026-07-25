// Single thread. Derive, for n floats:
//   out[0] = loads_float1  -- one scalar load instruction per element: n
//   out[1] = loads_float4  -- floor(n/4) vectorized float4 loads for the
//                             bulk, PLUS (n % 4) scalar loads for
//                             whatever tail doesn't fill a full float4
//   out[2] = loads_float1 / loads_float4  -- the instruction-count
//                             reduction factor from vectorizing
__global__ void float4_instr_counts(float* out, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // your code here
    }
}
