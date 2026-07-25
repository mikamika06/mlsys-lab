// BROKEN online softmax: updates the running max `m` but never rescales
// the running sum `l` -- every term already accumulated into `l` before
// this point was computed relative to the OLD (smaller) max, so it's
// now on the wrong scale relative to the new max, and every later
// division by `l` comes out wrong. Fix the accumulation so `l` is
// rescaled by exp(old_m - new_m) every time the max updates, BEFORE
// adding the new term (see gpu-fix-online-softmax's ref.cu for the
// exact formula).
__global__ void online_softmax(float* out, const float* in, int B, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < B) {
        float m = -1.0e30f;
        float l = 0.0f;
        int j = 0;
        while (j < N) {
            float x = in[row * N + j];
            float new_m = fmaxf(m, x);
            l = l + expf(x - new_m);  // BUG: missing l *= expf(m - new_m) rescale
            m = new_m;
            j = j + 1;
        }

        j = 0;
        while (j < N) {
            float x = in[row * N + j];
            out[row * N + j] = expf(x - m) / l;
            j = j + 1;
        }
    }
}
