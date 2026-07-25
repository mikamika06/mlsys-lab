// Reference: models a mixed-precision dot product the way a tensor core
// does it -- inputs quantized to a reduced-precision representation
// (this language subset has no real fp16/half2 type, so precision loss
// is modeled as rounding to the nearest multiple of a small quantum
// step, deterministically, with only arithmetic), multiplied and
// accumulated in full fp32.
//
// One warp (32 threads), n = 64 elements: thread `tid` is one "half2
// lane" -- it owns a PACKED PAIR of elements (i0 = 2*tid, i1 = i0+1),
// exactly the access pattern a real half2 load would produce (one load
// per lane fetches two logical values). Each lane quantizes and
// multiplies its own pair, sums its own 2 products, then a warp-shuffle
// tree combines all 32 lanes' partial sums into the final fp32 dot
// product.
__global__ void half2_matmul_dot(float* out, const float* a, const float* b, int n) {
    int tid = threadIdx.x;
    int i0 = tid * 2;
    int i1 = i0 + 1;

    float qstep = 1.0f / 256.0f;
    float a0 = floorf(a[i0] / qstep + 0.5f) * qstep;
    float b0 = floorf(b[i0] / qstep + 0.5f) * qstep;
    float a1 = floorf(a[i1] / qstep + 0.5f) * qstep;
    float b1 = floorf(b[i1] / qstep + 0.5f) * qstep;

    float val = a0 * b0 + a1 * b1;

    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    if (tid == 0) {
        out[0] = val;
    }
}
