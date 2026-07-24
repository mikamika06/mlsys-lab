#include "sol.hpp"
// #include <thread>   // uncomment if you parallelize over row blocks

// TODO: implement C = A * B for row-major matrices (see sol.hpp for layout).
// Optionally split the M rows into contiguous blocks and compute each block
// on its own std::thread (up to num_threads workers). The output must be
// identical for any num_threads >= 1.
//
// Right now it does nothing, so C stays zero and the gate fails.
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K, int num_threads) {
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K; (void)num_threads;
    // your code here
}
