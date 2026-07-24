// Reference: flatten the SIMT lane id into a (b, h, s) query-token owner
// using the same linear id formula as the token itself, so lane `lane`
// (for lane < total_tokens) owns exactly token `lane`. Lanes beyond
// total_tokens (the launch is warp-rounded, so there can be a few) store
// -1: idle, never a stray duplicate.
__global__ void map_tokens(int* out, int batch, int heads, int seq, int dim, int total_tokens) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane < total_tokens) {
        int per_batch = heads * seq;
        int b = lane / per_batch;
        int rem = lane - b * per_batch;
        int h = rem / seq;
        int s = rem - h * seq;
        int token = (b * heads + h) * seq + s;
        out[lane] = token;
    } else {
        out[lane] = -1;
    }
}
