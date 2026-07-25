// Pack a 16x16 A operand and a 16x8 B operand into the per-lane fragment
// layout a 32-lane warp holds them in for an m16n8k16-shaped MMA:
//   groupID = lane / 4  (0..7),  threadID_in_group = lane % 4  (0..3)
// A: each lane holds 8 elements, indices k in [0, 8):
//   half = k/4, sub = k%4; row = groupID + half*8; col = tid_in_group*4 + sub
// B: each lane holds 4 elements, indices k in [0, 4):
//   row = tid_in_group*4 + k; col = groupID
__global__ void pack_mma_fragment(float* fragA_out, float* fragB_out,
                                   const float* A, const float* B) {
    int lane = threadIdx.x;
    int groupID = lane / 4;
    int tid_in_group = lane % 4;
    // TODO: fill fragA_out[lane*8 .. lane*8+7] and fragB_out[lane*4 ..
    // lane*4+3] from A/B using the layout above.
}
