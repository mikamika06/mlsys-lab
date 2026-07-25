// Reference: shared-memory-tiled transpose. One block per (b,h) slice
// (blockIdx.x in [0, B*H)); each block has S*D=256 threads, one per
// element of that slice's S x D tile.
//
// Phase 1 loads the tile from `in` with THREAD tid reading in_row =
// tid/D, in_col = tid%D -- consecutive tid means consecutive in_col,
// which is exactly the fastest-varying axis of the (B,H,S,D) input
// layout, so the read is coalesced.
//
// Phase 2 writes to `out` with the SAME tid now interpreted as
// out_s = tid%S, out_d = tid/S -- consecutive tid means consecutive
// out_s, the fastest-varying axis of the (B,H,D,S) output layout, so
// the write is coalesced too. The actual "corner turn" (swapping which
// coordinate is fast-varying) happens entirely inside on-chip shared
// memory between the two phases.
__global__ void transpose_bhsd(float* out, const float* in, int B, int H, int S, int D) {
    int bh = blockIdx.x;
    int tid = threadIdx.x;

    int in_row = tid / D;
    int in_col = tid % D;
    int in_base = bh * S * D;

    __shared__ float tile[256];
    tile[in_row * D + in_col] = in[in_base + tid];
    __syncthreads();

    int out_s = tid % S;
    int out_d = tid / S;
    int out_base = bh * D * S;
    out[out_base + tid] = tile[out_s * D + out_d];
}
