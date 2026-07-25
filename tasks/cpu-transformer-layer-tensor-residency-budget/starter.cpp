#include "sol.hpp"

// TODO: for each tensor, find the smallest level L with tensor_bytes[i]
// <= cache_capacities[L] (residency_out[i] = L), or -1 if it exceeds
// every level (streamed). Sum tensor_bytes[i] for a resident tensor, or
// tensor_bytes[i]*num_uses[i] for a streamed one. See sol.hpp.
long classify_layer_residency(const long* tensor_bytes, const int* num_uses, int num_tensors,
                               const long* cache_capacities, int num_levels, int* residency_out) {
    (void)tensor_bytes; (void)num_uses; (void)cache_capacities; (void)num_levels;
    // your code here
    for (int i = 0; i < num_tensors; i++) residency_out[i] = 0;
    return 0;
}
