#include "sol.hpp"

// TODO: flops = 2*M*N*K. bytes = elem_bytes * ( M*K*(N/tile) + K*N*(M/tile)
// + M*N ). Return flops / bytes. See sol.hpp for the derivation.
double gemm_arithmetic_intensity(long M, long N, long K, long tile, long elem_bytes) {
    (void)M; (void)N; (void)K; (void)tile; (void)elem_bytes;
    // your code here
    return 0.0;
}
