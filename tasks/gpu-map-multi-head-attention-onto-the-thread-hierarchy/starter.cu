// Implement: flatten the SIMT lane id into a (b, h, s) query-token owner.
// lane = blockIdx.x * blockDim.x + threadIdx.x. For lane < total_tokens,
// decompose lane into (b, h, s) and store the flattened token id
// (b*heads + h)*seq + s. For lane >= total_tokens (the launch is
// warp-rounded, so there can be a few extra lanes), store -1.
__global__ void map_tokens(int* out, int batch, int heads, int seq, int dim, int total_tokens) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: decompose lane into (b, h, s) and store the token id, or -1.
    out[lane] = 0;
}
