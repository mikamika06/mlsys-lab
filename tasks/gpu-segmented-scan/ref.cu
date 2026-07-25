// Reference: inclusive segmented prefix sum over one warp of 32 lanes.
// `head_flag[i] == 1` marks the FIRST element of a new segment (element
// 0 is always a head); the scan restarts at every head instead of
// running across the whole warp.
//
// Alongside the running value `val`, every lane carries a running
// `flag`: has the growing window this lane's `val` has absorbed already
// crossed a segment head? Once it has, `val` stops absorbing anything
// further to the left (its own head boundary has been reached), but
// `flag` keeps propagating forward (via OR / max) so later, larger
// shuffle steps also know not to reach across that same boundary.
__global__ void segmented_scan(float* out, const float* in, const float* head_flag, int n) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    float val = in[tid];
    float flag = head_flag[tid];

    float val_up1 = __shfl_up_sync(0xffffffff, val, 1);
    float flag_up1 = __shfl_up_sync(0xffffffff, flag, 1);
    if (lane >= 1) {
        if (flag == 0.0f) { val = val + val_up1; }
        flag = fmaxf(flag, flag_up1);
    }

    float val_up2 = __shfl_up_sync(0xffffffff, val, 2);
    float flag_up2 = __shfl_up_sync(0xffffffff, flag, 2);
    if (lane >= 2) {
        if (flag == 0.0f) { val = val + val_up2; }
        flag = fmaxf(flag, flag_up2);
    }

    float val_up4 = __shfl_up_sync(0xffffffff, val, 4);
    float flag_up4 = __shfl_up_sync(0xffffffff, flag, 4);
    if (lane >= 4) {
        if (flag == 0.0f) { val = val + val_up4; }
        flag = fmaxf(flag, flag_up4);
    }

    float val_up8 = __shfl_up_sync(0xffffffff, val, 8);
    float flag_up8 = __shfl_up_sync(0xffffffff, flag, 8);
    if (lane >= 8) {
        if (flag == 0.0f) { val = val + val_up8; }
        flag = fmaxf(flag, flag_up8);
    }

    float val_up16 = __shfl_up_sync(0xffffffff, val, 16);
    float flag_up16 = __shfl_up_sync(0xffffffff, flag, 16);
    if (lane >= 16) {
        if (flag == 0.0f) { val = val + val_up16; }
        flag = fmaxf(flag, flag_up16);
    }

    out[tid] = val;
}
