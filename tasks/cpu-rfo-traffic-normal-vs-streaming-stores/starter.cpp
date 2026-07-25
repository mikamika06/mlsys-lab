#include "sol.hpp"

namespace {
long line_count(long total_bytes) {
    return (total_bytes + LINE_BYTES - 1) / LINE_BYTES;
}
} // namespace

long temporal_store_traffic(long total_bytes) {
    // your code here
    return line_count(total_bytes) * LINE_BYTES;
}

long nontemporal_store_traffic(long total_bytes) {
    // your code here
    return line_count(total_bytes) * LINE_BYTES;
}
