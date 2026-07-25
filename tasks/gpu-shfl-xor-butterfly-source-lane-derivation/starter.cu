// For each lane in [0, n), write the source lane that
// __shfl_xor_sync(mask_bits, val, mask) would read from -- lane XOR
// mask -- into out[lane]. `mask` is always a power of two (1, 2, 4, 8,
// or 16). This CUDA-C subset has no bitwise operators, so compute the
// XOR arithmetically: bit = (lane / mask) % 2; source = bit==0 ?
// lane+mask : lane-mask. See task.md.
__global__ void shfl_xor_source_lane(float* out, int mask, int n) {
    int lane = threadIdx.x;
    // TODO: out[lane] = lane XOR mask, computed arithmetically as
    // described above, guarded by lane < n.
}
