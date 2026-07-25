// Single thread. Compute arithmetic intensity for a decode step (batch=1
// token through a d_in x d_out linear layer) and a prefill step (batch=t
// tokens through the SAME weight matrix), then classify each against
// ridge = peak_flops / peak_bw.
__global__ void decode_prefill_ai(float* out, float d_in, float d_out, float t,
                                    float peak_flops, float peak_bw) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // your code here
    }
}
