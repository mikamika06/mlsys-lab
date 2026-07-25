// Reference: pack a 16x16 A operand and a 16x8 B operand into the
// per-lane "fragment" layout a 32-lane warp would hold them in for an
// m16n8k16-shaped tensor-core MMA (modeled after the real PTX-documented
// style of distributing an operand across a warp via groupID/
// threadID_in_group -- not claimed to be bit-identical to any specific
// real ISA revision, since real MMA execution is outside what this
// software GPU models).
//
//   groupID        = lane / 4   (0..7)
//   threadID_in_group = lane % 4   (0..3)
//
// A (16 rows x 16 cols): each lane holds 8 elements. For k in [0, 8):
//   half = k / 4;  sub = k % 4;
//   row = groupID + half * 8;             (covers all 16 rows: 8 groups x 2 halves)
//   col = threadID_in_group * 4 + sub;    (covers all 16 cols: 4 lanes x 4 sub each)
//
// B (16 rows x 8 cols): each lane holds 4 elements. For k in [0, 4):
//   row = threadID_in_group * 4 + k;      (covers all 16 rows: 4 lanes x 4 k each)
//   col = groupID;                        (covers all 8 cols: one group per col)
__global__ void pack_mma_fragment(float* fragA_out, float* fragB_out,
                                   const float* A, const float* B) {
    int lane = threadIdx.x;
    int groupID = lane / 4;
    int tid_in_group = lane % 4;

    for (int k = 0; k < 8; k = k + 1) {
        int half = k / 4;
        int sub = k % 4;
        int row = groupID + half * 8;
        int col = tid_in_group * 4 + sub;
        fragA_out[lane * 8 + k] = A[row * 16 + col];
    }

    for (int k = 0; k < 4; k = k + 1) {
        int row = tid_in_group * 4 + k;
        int col = groupID;
        fragB_out[lane * 4 + k] = B[row * 8 + col];
    }
}
