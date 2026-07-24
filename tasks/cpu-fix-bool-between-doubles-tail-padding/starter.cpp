#include "sol.hpp"
#include <cstddef>

// BROKEN: each bool is wedged directly after a double, forcing the
// compiler to insert alignment padding before the NEXT double every time
// (and again at the very end) instead of just once.
struct BadRecord {
    double a;
    bool flag1;
    double b;
    bool flag2;
};

size_t record_size() { return sizeof(BadRecord); }
size_t offset_a() { return offsetof(BadRecord, a); }
size_t offset_b() { return offsetof(BadRecord, b); }
size_t offset_flag1() { return offsetof(BadRecord, flag1); }
size_t offset_flag2() { return offsetof(BadRecord, flag2); }
