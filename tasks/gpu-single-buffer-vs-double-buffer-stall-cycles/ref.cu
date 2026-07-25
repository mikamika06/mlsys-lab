// Reference: T tiles, each needing `load_cycles` to fetch and
// `compute_cycles` to process.
//   single buffer: only one buffer exists, so every tile's load and
//   compute happen strictly back to back -- nothing can overlap.
//   total = T * (load_cycles + compute_cycles).
//   double buffer: tile 0's load can't overlap anything (prologue,
//   `load_cycles`). From then on, tile t's load overlaps tile t-1's
//   compute -- each steady-state step costs max(load_cycles,
//   compute_cycles), for T-1 steps. The very last tile's compute has
//   nothing left to overlap with (epilogue, `compute_cycles`).
//   total = load_cycles + (T-1)*max(load_cycles,compute_cycles) + compute_cycles.
__global__ void buffering_cycles(int T, int load_cycles, int compute_cycles, float* out) {
    float single_total = T * (load_cycles + compute_cycles) + 0.0;

    float mx = load_cycles > compute_cycles ? load_cycles : compute_cycles;
    float double_total = load_cycles + (T - 1) * mx + compute_cycles;

    out[0] = single_total;
    out[1] = double_total;
}
