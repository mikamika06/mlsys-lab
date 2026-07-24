#include "sol.hpp"

long long gemmFlops(const GemmConfig& cfg) {
    long long perElem = (long long)cfg.K + ((long long)cfg.K - 1) + 1;  // dot product + alpha mult
    if (cfg.beta != 0.0) perElem += 2;                                  // beta mult + add
    return (long long)cfg.M * (long long)cfg.N * perElem;
}
