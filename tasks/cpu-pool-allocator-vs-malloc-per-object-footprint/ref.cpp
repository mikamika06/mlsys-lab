#include "sol.hpp"

namespace {
long round_up(long x, long a) {
    return ((x + a - 1) / a) * a;
}
} // namespace

long malloc_per_object_footprint(int count, int obj_bytes) {
    long per_object = round_up(HEADER_BYTES + obj_bytes, ALIGN_BYTES);
    return static_cast<long>(count) * per_object;
}

long pool_footprint(int count, int obj_bytes) {
    long raw = HEADER_BYTES + static_cast<long>(count) * obj_bytes;
    return round_up(raw, ALIGN_BYTES);
}

double footprint_ratio(int count, int obj_bytes) {
    return static_cast<double>(malloc_per_object_footprint(count, obj_bytes)) /
           static_cast<double>(pool_footprint(count, obj_bytes));
}
