// BUG: this kernel assumes the warp automatically "reconverges" after
// the divergent `if` block, so it's safe to shuffle right after with a
// full 0xffffffff mask. That guarantee doesn't hold here (and doesn't
// hold on real Volta+ hardware with independent thread scheduling
// either): the __syncthreads() inside the `if` only synchronizes the
// lanes that TOOK that branch, so lanes 0..15 arrive at the shuffle a
// full round later than lanes 16..31 do, and the shuffle reads whatever
// its neighbor lane has published *in its own round* -- which, for the
// lane sitting right at the branch boundary, is nothing yet. Fix it so
// every lane reaches the same synchronization point before any lane
// reaches the shuffle. See task.md.
__global__ void divergent_shuffle(float* out, const float* in, int n) {
    int tid = threadIdx.x;
    float val = in[tid];
    if (tid < 16) {
        val = val * 2.0;
        __syncthreads();
    }
    float shuffled = __shfl_up_sync(0xffffffff, val, 1);
    out[tid] = shuffled;
}
