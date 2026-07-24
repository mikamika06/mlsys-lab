#include "sol.hpp"

double gemm_arithmetic_intensity(long M, long N, long K, long tile, long elem_bytes) {
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double a_bytes = (double)elem_bytes * (double)M * (double)K * ((double)N / (double)tile);
    double b_bytes = (double)elem_bytes * (double)K * (double)N * ((double)M / (double)tile);
    double c_bytes = (double)elem_bytes * (double)M * (double)N;
    double bytes = a_bytes + b_bytes + c_bytes;
    return flops / bytes;
}
