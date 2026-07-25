// Reference (single thread): the SAME linear layer (d_in x d_out weight
// matrix), run two ways.
//   decode (batch=1, one token): flops = 2*d_in*d_out (one matrix-vector
//   product), bytes = 4*(d_in*d_out + d_in + d_out) -- the weight matrix
//   dominates the byte traffic and is used for exactly ONE output
//   vector, so AI approaches a FIXED ~0.5 FLOPs/byte no matter how big
//   d_in/d_out are.
//   prefill (batch=t tokens): flops = 2*t*d_in*d_out (one matrix-matrix
//   product), bytes = 4*(d_in*d_out + t*d_in + t*d_out) -- the SAME
//   weight matrix is loaded once and reused across all t tokens, so AI
//   grows with t.
__global__ void decode_prefill_ai(float* out, float d_in, float d_out, float t,
                                    float peak_flops, float peak_bw) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float ridge = peak_flops / peak_bw;

        float decode_flops = 2.0f * d_in * d_out;
        float decode_bytes = 4.0f * (d_in * d_out + d_in + d_out);
        float decode_ai = decode_flops / decode_bytes;

        float prefill_flops = 2.0f * t * d_in * d_out;
        float prefill_bytes = 4.0f * (d_in * d_out + t * d_in + t * d_out);
        float prefill_ai = prefill_flops / prefill_bytes;

        out[0] = decode_ai;
        out[1] = prefill_ai;
        out[2] = (decode_ai >= ridge) ? 1.0f : 0.0f;
        out[3] = (prefill_ai >= ridge) ? 1.0f : 0.0f;
    }
}
