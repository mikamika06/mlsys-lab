#include "sol.hpp"

// TODO: walk the fields, tracking the running offset and the max field
// alignment seen so far; insert inter-field padding before each field so
// its offset is a multiple of its alignment; after the last field, pad
// the total size up to a multiple of the max alignment (tail padding);
// return (padded_size - sum_of_field_sizes) * count. See sol.hpp.
long total_padding_bytes(const int* field_sizes, const int* field_aligns, int num_fields, long count) {
    (void)field_sizes; (void)field_aligns; (void)num_fields; (void)count;
    // your code here
    return 0;
}
