// TODO: solve for the largest power-of-two tile size T such that the
// four Q/K/V/O tiles (each T x head_dim floats) fit in sram_bytes:
// T <= sram_bytes / (16 * head_dim), rounded down to a power of two.
// See ref.cu's approach: floor the raw bound, then floor its log2 (with
// a tiny epsilon guard) and raise 2 to that power. If the raw bound is
// below 1, the answer is 0 (nothing fits).
__global__ void derive_tile_size(int sram_bytes, int head_dim, float* out) {
    out[0] = 0.0;
}
