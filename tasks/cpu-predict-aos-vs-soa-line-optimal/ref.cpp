#include <set>
#include <vector>
#include "sol.hpp"

namespace {
void mark_lines(std::set<long>& lines, long addr, long size, int line_bytes) {
    long first = addr / line_bytes;
    long last = (addr + size - 1) / line_bytes;
    for (long l = first; l <= last; l++) lines.insert(l);
}
}  // namespace

int soa_is_optimal(int N, int F, const int* field_bytes, const bool* mask) {
    const int LINE = 64;

    // AoS: fields packed back-to-back per record.
    std::vector<long> aos_offset(F);
    long record_bytes = 0;
    for (int f = 0; f < F; f++) {
        aos_offset[f] = record_bytes;
        record_bytes += field_bytes[f];
    }

    std::set<long> aos_lines;
    for (int r = 0; r < N; r++) {
        for (int f = 0; f < F; f++) {
            if (!mask[f]) continue;
            long addr = (long)r * record_bytes + aos_offset[f];
            mark_lines(aos_lines, addr, field_bytes[f], LINE);
        }
    }

    // SoA: each field gets its own array, base padded to a whole number
    // of lines so fields never share a line.
    std::vector<long> soa_base(F);
    long next_base = 0;
    for (int f = 0; f < F; f++) {
        soa_base[f] = next_base;
        long bytes = (long)N * field_bytes[f];
        long lines = (bytes + LINE - 1) / LINE;
        next_base += lines * LINE;
    }

    std::set<long> soa_lines;
    for (int f = 0; f < F; f++) {
        if (!mask[f]) continue;
        for (int r = 0; r < N; r++) {
            long addr = soa_base[f] + (long)r * field_bytes[f];
            mark_lines(soa_lines, addr, field_bytes[f], LINE);
        }
    }

    return (soa_lines.size() <= aos_lines.size()) ? 1 : 0;
}
