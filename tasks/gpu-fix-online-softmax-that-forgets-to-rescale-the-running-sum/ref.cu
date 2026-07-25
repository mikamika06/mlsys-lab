// Online (streaming) softmax: one thread per row, single running pass
// over the row to build a running max `m` and running sum `l` of
// exp(x - m), THEN a second pass to write the normalized output. The
// safety trick: every time a new element pushes the running max up, `l`
// -- built against the OLD max -- must be rescaled by exp(old_m - new_m)
// before adding the new term, so every term in the running sum stays
// expressed relative to the CURRENT running max.
__global__ void online_softmax(float* out, const float* in, int B, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < B) {
        float m = -1.0e30f;
        float l = 0.0f;
        int j = 0;
        while (j < N) {
            float x = in[row * N + j];
            float new_m = fmaxf(m, x);
            l = l * expf(m - new_m) + expf(x - new_m);
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
