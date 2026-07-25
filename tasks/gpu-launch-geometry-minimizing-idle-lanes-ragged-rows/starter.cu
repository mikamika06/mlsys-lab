// Ragged rows, COMPACTED (not padded): row r's lens[r] real elements are
// concatenated back-to-back in `data`. tid = blockIdx.x*blockDim.x +
// threadIdx.x, guarded by tid < total. Find tid's row by scanning
// offsets[0..R): offsets[r] = sum of lens[0..r) (the flat start of row
// r); row is the LAST r for which offsets[r] <= tid (no `break` in this
// dialect -- just keep overwriting `row` as you scan every r). Then
// out[tid] = data[tid] + row_scale[row].
__global__ void ragged_process(float* out, const float* data, const int* offsets,
                                const float* row_scale, int R, int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // your code here
}
