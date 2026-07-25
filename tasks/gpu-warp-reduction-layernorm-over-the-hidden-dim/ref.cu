// Reference: one warp (32 threads) per row, D=128 (4 elements per
// lane, strided by 32). Each lane accumulates a PARTIAL sum and
// sum-of-squares over its own 4 elements, then a butterfly XOR-shuffle
// all-reduce (offsets 16,8,4,2,1) combines all 32 lanes' partials --
// every lane ends up holding the row's FULL sum/sumsq, no separate
// broadcast step needed. Each lane then normalizes its own 4 elements
// using that shared mean/variance.
__global__ void warp_layernorm(const float* x, const float* gamma, const float* beta,
                                float* y, int rows, int D, float eps) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = global_tid / 32;
    int lane = global_tid % 32;
    if (row < rows) {
        float sum = 0.0;
        float sumsq = 0.0;
        int c = 0;
        while (c < D) {
            int d = lane + c;
            float v = x[row * D + d];
            sum = sum + v;
            sumsq = sumsq + v * v;
            c = c + 32;
        }

        float t16 = __shfl_xor_sync(0xffffffff, sum, 16);
        sum = sum + t16;
        float t8 = __shfl_xor_sync(0xffffffff, sum, 8);
        sum = sum + t8;
        float t4 = __shfl_xor_sync(0xffffffff, sum, 4);
        sum = sum + t4;
        float t2 = __shfl_xor_sync(0xffffffff, sum, 2);
        sum = sum + t2;
        float t1 = __shfl_xor_sync(0xffffffff, sum, 1);
        sum = sum + t1;

        float q16 = __shfl_xor_sync(0xffffffff, sumsq, 16);
        sumsq = sumsq + q16;
        float q8 = __shfl_xor_sync(0xffffffff, sumsq, 8);
        sumsq = sumsq + q8;
        float q4 = __shfl_xor_sync(0xffffffff, sumsq, 4);
        sumsq = sumsq + q4;
        float q2 = __shfl_xor_sync(0xffffffff, sumsq, 2);
        sumsq = sumsq + q2;
        float q1 = __shfl_xor_sync(0xffffffff, sumsq, 1);
        sumsq = sumsq + q1;

        float mean = sum / D;
        float var = sumsq / D - mean * mean;
        float invstd = 1.0 / sqrtf(var + eps);

        int c2 = 0;
        while (c2 < D) {
            int d = lane + c2;
            float v = x[row * D + d];
            y[row * D + d] = (v - mean) * invstd * gamma[d] + beta[d];
            c2 = c2 + 32;
        }
    }
}
