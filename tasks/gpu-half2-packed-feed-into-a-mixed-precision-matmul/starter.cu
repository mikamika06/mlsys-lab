// One warp (32 threads), n = 64 elements. Thread tid owns a packed pair
// (i0 = 2*tid, i1 = i0+1) -- a modeled "half2 lane". Quantize each of
// a[i0], b[i0], a[i1], b[i1] to the nearest multiple of qstep =
// 1.0f/256.0f (floorf(x/qstep + 0.5f) * qstep), multiply-accumulate the
// pair in fp32, then warp-shuffle-reduce all 32 lanes' partial sums into
// out[0].
__global__ void half2_matmul_dot(float* out, const float* a, const float* b, int n) {
    int tid = threadIdx.x;
    int i0 = tid * 2;
    int i1 = i0 + 1;
    // your code here
}
