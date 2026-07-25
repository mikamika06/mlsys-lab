// Compute an inclusive SEGMENTED prefix sum over one warp of 32
// elements: out[i] = sum of in[k] for k from this element's segment's
// head up through i. head_flag[i] == 1 marks the first element of a new
// segment (head_flag[0] is always 1). Use the 5-step shfl_up ladder
// (delta 1,2,4,8,16), but ALSO shuffle a running `flag` alongside `val`:
// only merge val_up into val if this lane's flag is still 0 (hasn't
// crossed its own segment head yet); flag itself always propagates
// forward via max(flag, flag_up) once lane >= delta. See task.md.
__global__ void segmented_scan(float* out, const float* in, const float* head_flag, int n) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    float val = in[tid];
    float flag = head_flag[tid];
    // TODO: 5-step ladder, each step reading val_up/flag_up via
    // __shfl_up_sync, merging val only when flag == 0, then updating
    // flag = fmaxf(flag, flag_up).
    out[tid] = val;
}
