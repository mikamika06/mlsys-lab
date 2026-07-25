// Compute row-wise softmax: out[i,:] = exp(logits[i,:]) / sum(exp(logits[i,:])),
// one thread per row (i = threadIdx.x, n_rows rows of D logits each).
// Subtract the row's own max logit before calling expf() so no exponent
// ever overflows, regardless of how large the logits are -- the ratio is
// unchanged by that constant shift. See task.md.
__global__ void safe_softmax_row(float* out, const float* logits, int n_rows, int D) {
    int i = threadIdx.x;
    // TODO: find row max; sum expf(logit - max) over the row; write
    // expf(logit - max) / sum for every element.
}
