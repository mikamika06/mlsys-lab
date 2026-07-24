#include "sol.hpp"

static int type_size(FieldType t) {
    switch (t) {
        case FieldType::Bool:     return 1;
        case FieldType::Char:     return 1;
        case FieldType::Short:    return 2;
        case FieldType::Int:      return 4;
        case FieldType::Long:     return 8;
        case FieldType::LongLong: return 8;
        case FieldType::Float:    return 4;
        case FieldType::Double:   return 8;
        case FieldType::Pointer:  return 8;
    }
    return 0;
}

// BUG: inter-field padding is correct, but the trailing tail padding (to
// round the whole class up to its strictest member's alignment) is missing.
int compute_layout(const FieldType* fields, int n, int* out_offsets) {
    int off = 0, maxa = 1;
    for (int i = 0; i < n; ++i) {
        int s = type_size(fields[i]);
        int a = s;
        off += (a - (off % a)) % a;   // inter-field alignment padding
        out_offsets[i] = off;
        off += s;
        if (a > maxa) maxa = a;
    }
    // TODO: missing tail padding — `off` is not rounded up to `maxa` here.
    return off;
}
