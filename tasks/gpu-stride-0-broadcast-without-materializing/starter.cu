// Broadcast-add a length-c bias vector across every row of an r x c
// row-major matrix `a` (n = r*c total elements), WITHOUT assuming `bias`
// was ever expanded to n elements -- it only has c real elements. Every
// row must re-read the SAME c bias values.
__global__ void broadcast_add(float* out, const float* a, const float* bias,
                               int r, int c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // TODO: out[i] = a[i] + bias[i % c];
    }
}
