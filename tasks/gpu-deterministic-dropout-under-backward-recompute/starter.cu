// Dropout with NO stored mask: the forward pass decides keep/drop for
// element i from a pure hash of (seed, i) and scales by 1/keep_prob; the
// backward pass must RECOMPUTE that exact same hash from (seed, i) --
// using the global index i, not just threadIdx.x -- to reproduce the
// identical mask for grad_in, with no mask buffer passed between them.
__global__ void dropout_fwd_bwd(float* fwd_out, float* bwd_grad, const float* x,
                                 const float* grad_in, float seed, float keep_prob, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard i < n. Compute h = i, then 3 rounds of
    // h = (h*48271 + seed + r*7919) % 1000003 to get rand01 = h/1000003.
    // keep = rand01 < keep_prob. fwd_out[i] = keep * x[i] / keep_prob.
    // Recompute the SAME hash again for bwd_grad[i] = keep * grad_in[i] / keep_prob.
}
