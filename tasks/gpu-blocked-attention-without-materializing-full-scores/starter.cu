// Implement single-pass (online-softmax) attention: thread i owns query
// row i and must stream over all S keys/values, maintaining a running
// max, running sum, and running (unnormalized) output accumulator --
// without ever materializing the full S x S score matrix. See ref
// description in the task for the exact update rule.
__global__ void flash_attn(float* out, const float* Q, const float* K, const float* V, int S, int d, int bk) {
    int i = threadIdx.x;
    // TODO: stream over K/V (in chunks of `bk`, or one at a time) and
    // accumulate softmax(Q[i] . K^T / sqrt(d)) @ V into out[i*d .. i*d+d)
    // using the online-softmax update rule (rescale on every new max).
}
