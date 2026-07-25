#include "sol.hpp"

namespace {
long line_count(long bytes) { return bytes / LINE_BYTES; }
}  // namespace

long modeled_dram_traffic(long h_bytes, long a_bytes, bool use_nontemporal) {
    reset_cache();
    long traffic = 0;

    long h_lines = line_count(h_bytes);
    long a_lines = line_count(a_bytes);

    // Step 1: warm H.
    for (long k = 0; k < h_lines; k++) {
        long addr = k * LINE_BYTES;
        if (touch_byte(addr)) traffic += LINE_BYTES;
    }

    // Step 2: write A.
    for (long k = 0; k < a_lines; k++) {
        long addr = h_bytes + k * LINE_BYTES;
        if (use_nontemporal) {
            nontemporal_store(addr);
            traffic += LINE_BYTES;
        } else {
            if (touch_byte(addr)) traffic += 2 * LINE_BYTES;
        }
    }

    // Step 3: re-touch H.
    for (long k = 0; k < h_lines; k++) {
        long addr = k * LINE_BYTES;
        if (touch_byte(addr)) traffic += LINE_BYTES;
    }

    return traffic;
}
