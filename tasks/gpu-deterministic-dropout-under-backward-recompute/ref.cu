// Reference: dropout with NO stored mask. The forward pass computes each
// element's keep/drop decision from a pure counter-based hash of
// (seed, i) and applies it; the backward pass RECOMPUTES the identical
// hash from the same (seed, i) -- never reading anything the forward
// pass wrote -- to reproduce the exact same mask for the gradient.
__global__ void dropout_fwd_bwd(float* fwd_out, float* bwd_grad, const float* x,
                                 const float* grad_in, float seed, float keep_prob, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float h = i;
        for (int r = 0; r < 3; r++) {
            h = (h * 48271.0f + seed + r * 7919.0f) % 1000003.0f;
        }
        float rand01 = h / 1000003.0f;
        float keep = (rand01 < keep_prob) ? 1.0f : 0.0f;
        fwd_out[i] = keep * x[i] / keep_prob;

        float h2 = i;
        for (int r = 0; r < 3; r++) {
            h2 = (h2 * 48271.0f + seed + r * 7919.0f) % 1000003.0f;
        }
        float rand01_2 = h2 / 1000003.0f;
        float keep2 = (rand01_2 < keep_prob) ? 1.0f : 0.0f;
        bwd_grad[i] = keep2 * grad_in[i] / keep_prob;
    }
}
