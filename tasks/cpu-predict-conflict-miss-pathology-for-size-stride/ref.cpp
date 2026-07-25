#include "sol.hpp"
#include <numeric>

int classify_pathological(long array_size, long stride) {
    long d = (stride / LINE_BYTES) % NUM_SETS;
    long g = std::gcd(d, static_cast<long>(NUM_SETS)); // gcd(0, S) == S, matching "1 distinct set"
    long distinct_sets = NUM_SETS / g;
    long n = array_size / stride;
    long lines_per_busiest_set = n / distinct_sets;
    return (lines_per_busiest_set > WAYS) ? 1 : 0;
}
