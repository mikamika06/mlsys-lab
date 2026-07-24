#include "sol.hpp"

// TODO: partition fields[0..n) into hot_out/cold_out by is_hot[i], then
// reorder hot_out (descending by size is optimal here, since alignment ==
// size for every field kind used) to minimize struct_size(hot_out,
// hot_count). See sol.hpp for the exact contract.
int split_struct(const int* fields, const int* is_hot, int n,
                  int* hot_out, int* cold_out) {
    (void)fields; (void)is_hot; (void)n; (void)hot_out; (void)cold_out;
    // your code here
    return 0;
}
