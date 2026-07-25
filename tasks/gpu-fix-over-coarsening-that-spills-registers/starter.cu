// Each thread coarsens over 8 elements (base .. base+7): out[i] = in[i]^2 + c.
//
// BUG: fully unrolled by hand into 16 separately named temporaries (v0..v7,
// r0..r7), all of them live across most of the function body instead of
// being reused one at a time. Same arithmetic, same result -- but this
// needs 16+ simultaneously-live per-thread values where the loop-based
// version below only ever needs one. Rewrite it as a loop with a single
// reused temporary.
__global__ void coarsened_square(float* out, const float* in, int n, float c) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    float v0 = in[base + 0];
    float v1 = in[base + 1];
    float v2 = in[base + 2];
    float v3 = in[base + 3];
    float v4 = in[base + 4];
    float v5 = in[base + 5];
    float v6 = in[base + 6];
    float v7 = in[base + 7];
    float r0 = v0 * v0 + c;
    float r1 = v1 * v1 + c;
    float r2 = v2 * v2 + c;
    float r3 = v3 * v3 + c;
    float r4 = v4 * v4 + c;
    float r5 = v5 * v5 + c;
    float r6 = v6 * v6 + c;
    float r7 = v7 * v7 + c;
    out[base + 0] = r0;
    out[base + 1] = r1;
    out[base + 2] = r2;
    out[base + 3] = r3;
    out[base + 4] = r4;
    out[base + 5] = r5;
    out[base + 6] = r6;
    out[base + 7] = r7;
}
