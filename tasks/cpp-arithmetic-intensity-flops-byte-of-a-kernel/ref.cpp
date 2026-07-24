#include "sol.hpp"
#include <limits>

double arithmetic_intensity(int struct_bytes,
                             const int* field_bytes, int num_fields,
                             const int* reads, int num_reads,
                             const int* writes, int num_writes,
                             int flops, bool is_aos) {
    (void)num_fields;
    long long bytes = 0;
    if (is_aos) {
        if (num_reads > 0)  bytes += struct_bytes;
        if (num_writes > 0) bytes += struct_bytes;
    } else {
        for (int i = 0; i < num_reads; i++)  bytes += field_bytes[reads[i]];
        for (int i = 0; i < num_writes; i++) bytes += field_bytes[writes[i]];
    }
    if (bytes == 0) return std::numeric_limits<double>::infinity();
    return (double)flops / (double)bytes;
}
