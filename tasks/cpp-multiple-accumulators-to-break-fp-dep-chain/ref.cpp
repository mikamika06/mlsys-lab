#include "sol.hpp"

// Reference: K independent (strided) accumulators, then combine.
double reduce_multi_acc(const double* x, int n, double* partial, int K) {
    for (int j = 0; j < K; j++) partial[j] = 0.0;
    for (int i = 0; i < n; i++) partial[i % K] += x[i];
    double total = 0.0;
    for (int j = 0; j < K; j++) total += partial[j];
    return total;
}
