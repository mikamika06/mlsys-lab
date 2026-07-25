// Reference: two transpose strategies for the SAME 16x16 matrix, in one
// file (this CUDA-C subset allows several distinctly-named kernels per
// source, just not two with the same name).
//
// transpose_in_place takes a SINGLE pointer -- there is no second buffer
// to write into, by construction of its own signature. Only the upper-
// triangle threads (row < col) do anything: each one swaps its element
// with its mirror across the diagonal. Diagonal threads (row == col) and
// lower-triangle threads (row > col) do nothing -- if every thread swapped
// unconditionally, each pair would be swapped twice and undo itself.
//
// transpose_out_of_place takes two pointers and never touches `in`: every
// thread writes its element to the mirrored position in a full second
// n*n-element buffer.
__global__ void transpose_in_place(float* A, int n) {
    int tid = threadIdx.x;
    int row = tid / n;
    int col = tid % n;
    if (row < col) {
        float tmp = A[row * n + col];
        A[row * n + col] = A[col * n + row];
        A[col * n + row] = tmp;
    }
}

__global__ void transpose_out_of_place(float* out, const float* in, int n) {
    int tid = threadIdx.x;
    int row = tid / n;
    int col = tid % n;
    out[col * n + row] = in[row * n + col];
}
