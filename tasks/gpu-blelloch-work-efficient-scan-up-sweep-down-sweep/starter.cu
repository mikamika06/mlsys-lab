// Compute the EXCLUSIVE prefix sum of the 32 input elements via the
// work-efficient (Blelloch) up-sweep/down-sweep algorithm:
//   1. Load in[tid] into shared memory temp[tid].
//   2. Up-sweep: for d = n/2, n/4, ..., 1 (halving each step), the
//      first d threads add temp[ai] into temp[bi], where
//      ai = offset*(2*tid+1)-1, bi = offset*(2*tid+2)-1, offset
//      starting at 1 and doubling after every step. __syncthreads()
//      before each step (some threads read what others wrote last step).
//   3. Set temp[n-1] = 0 (thread 0 only).
//   4. Down-sweep: for d = 1, 2, ..., n/2 (doubling each step, offset
//      halving first), the first d threads swap temp[ai] and temp[bi]
//      then add the old temp[ai] into the new temp[bi].
//   5. __syncthreads(), then out[tid] = temp[tid].
__global__ void scan(float* out, const float* in, int n) {
    __shared__ float temp[32];
    int tid = threadIdx.x;
    temp[tid] = in[tid];
    __syncthreads();
    // your code here
}
