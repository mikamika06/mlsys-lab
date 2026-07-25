// TODO: out[0] = single-buffered total = T * (load_cycles +
// compute_cycles) (no overlap possible). out[1] = double-buffered
// total = load_cycles (prologue) + (T-1)*max(load_cycles,
// compute_cycles) (steady state, one tile's load overlapping the
// previous tile's compute) + compute_cycles (epilogue, the last
// tile's compute has nothing left to overlap). See ref.cu (note the
// "+ 0.0" there forces float arithmetic).
__global__ void buffering_cycles(int T, int load_cycles, int compute_cycles, float* out) {
    out[0] = 0.0;
    out[1] = 0.0;
}
