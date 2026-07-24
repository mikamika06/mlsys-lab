#pragma once

// C = alpha * A * B + beta * C, where A is MxK, B is KxN, C is MxN.
struct GemmConfig {
    int M;
    int N;
    int K;
    double alpha;
    double beta;
};

// Computes the EXACT FLOP count for cfg, under this strict counting model.
// Each of the M*N output elements costs:
//   - K multiplications and (K - 1) additions for the dot product
//   - +1 multiplication for the alpha scale
//   - if cfg.beta != 0.0: +1 multiplication (beta * C_ij) and +1 addition
//     (adding it to the alpha term). If cfg.beta == 0.0, BLAS optimizes
//     this whole term away -- the alpha term directly overwrites C_ij, so
//     there is NO beta multiplication and NO extra addition.
// Return the total over all M*N elements.
long long gemmFlops(const GemmConfig& cfg);
