#include "sol.hpp"
#include <cstddef>

// Fixed: both doubles together (naturally 8-byte aligned back-to-back, no
// gaps), then both bools together (1 byte each, no alignment needs) --
// only one rounding gap at the very end, to bring the struct up to its own
// 8-byte alignment.
struct FixedRecord {
    double a;
    double b;
    bool flag1;
    bool flag2;
};

size_t record_size() { return sizeof(FixedRecord); }
size_t offset_a() { return offsetof(FixedRecord, a); }
size_t offset_b() { return offsetof(FixedRecord, b); }
size_t offset_flag1() { return offsetof(FixedRecord, flag1); }
size_t offset_flag2() { return offsetof(FixedRecord, flag2); }
