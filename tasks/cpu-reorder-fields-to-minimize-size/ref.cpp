#include "sol.hpp"

namespace {
struct PackedStruct {
    double value_x;
    int64_t id;
    int32_t count;
    int16_t small_val;
    bool flag_a;
    char tag;
    bool flag_b;
};
}  // namespace

size_t packed_struct_size() { return sizeof(PackedStruct); }
