#include "sol.hpp"

long total_padding_bytes(const int* field_sizes, const int* field_aligns, int num_fields, long count) {
    if (num_fields <= 0 || count <= 0) return 0;

    long offset = 0;
    long max_align = 1;
    long sum_sizes = 0;
    for (int i = 0; i < num_fields; i++) {
        long a = field_aligns[i];
        long s = field_sizes[i];
        sum_sizes += s;
        if (a > max_align) max_align = a;
        long rem = offset % a;
        if (rem != 0) offset += (a - rem);  // inter-field padding before this field
        offset += s;
    }
    long rem = offset % max_align;
    long padded_size = offset + (rem != 0 ? (max_align - rem) : 0);  // tail padding

    long padding_per_record = padded_size - sum_sizes;
    return padding_per_record * count;
}
