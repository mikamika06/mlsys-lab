// Reference: broadcast-add a length-c bias vector across every row of an
// r x c row-major matrix WITHOUT ever materializing an expanded r*c copy
// of it. `bias` only has c real elements; the broadcast is expressed by
// indexing it with `i % c` (every row re-reads the same c values -- a
// "stride 0" repeat across the row dimension), not by assuming a
// pre-expanded buffer the same size as `a`.
__global__ void broadcast_add(float* out, const float* a, const float* bias,
                               int r, int c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + bias[i % c];
    }
}
