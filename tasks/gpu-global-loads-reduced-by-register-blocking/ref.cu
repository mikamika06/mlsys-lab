// Reference: for an M x N output tile computed against a depth-K
// contraction, register blocking (thread coarsening) has each thread
// compute C adjacent output columns instead of 1. That means M*(N/C)
// threads total, each looping over K -- and at each k, each thread
// loads its ROW value from A exactly ONCE (a register), reusing it for
// all C of its outputs, while it still needs a fresh COLUMN value from
// B for every one of its C outputs (B doesn't repeat across outputs the
// way A does). A's total load count therefore shrinks by a factor of C
// versus giving every output its own thread (C=1); B's total load
// count is unaffected by C entirely.
__global__ void derive_loads(int M, int N, int K, int C, float* out) {
    int threads = M * (N / C);
    out[0] = threads * K;  // A loads, with register blocking
    out[1] = M * N * K;    // B loads (always -- one fresh load per output per k)
}
