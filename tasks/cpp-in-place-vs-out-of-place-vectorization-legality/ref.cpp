#include "sol.hpp"

namespace {
constexpr int N = 16;
}

// Reference: the exemptions, then a pairwise byte-range-overlap check
// (equivalent to, but cheaper than, building the full write/read byte-address
// sets from task.md's definition -- two contiguous ranges [a0,a1) and
// [b0,b1) overlap iff a0 < b1 && b0 < a1).
bool is_safe_to_vectorize(const KernelSpec& spec) {
    if (spec.has_restrict) return true;

    long src_base = (long)spec.src_offset + (long)spec.src_elem_shift * spec.src_size;
    long dest_base = (long)spec.dest_offset + (long)spec.dest_elem_shift * spec.dest_size;

    if (src_base == dest_base && spec.src_size == spec.dest_size) return true;

    int V = spec.vector_width;
    for (int b = 0; b < N / V; b++) {
        for (int i = b * V; i < (b + 1) * V; i++) {
            long d_start = (long)i * spec.struct_size + dest_base;
            long d_end = d_start + spec.dest_size;
            for (int j = (b + 1) * V; j < N; j++) {
                long s_start = (long)j * spec.struct_size + src_base;
                long s_end = s_start + spec.src_size;
                if (d_start < s_end && s_start < d_end) return false;  // RAW hazard
            }
        }
    }
    return true;
}
