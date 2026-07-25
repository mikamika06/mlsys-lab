// Reference: FlashAttention-style fused attention tiles Q, K, V (and an
// output accumulator O) into shared memory per block. With square
// tiling (block row count Br == block column count Bc == T), the four
// tiles -- Q and O sized T x head_dim, K and V also sized T x head_dim
// -- together need 4 * T * head_dim * 4 bytes (float32) of shared
// memory. Solve for the LARGEST power-of-two T that still fits the
// budget: T <= sram_bytes / (16 * head_dim), rounded down to a power of
// two (real kernels want block sizes that are powers of two for warp
// alignment).
__global__ void derive_tile_size(int sram_bytes, int head_dim, float* out) {
    float raw = floorf(sram_bytes / (16.0 * head_dim));
    if (raw < 1.0) {
        out[0] = 0.0;
    } else {
        float e = floorf(logf(raw) / logf(2.0) + 0.000000001);
        out[0] = powf(2.0, e);
    }
}
