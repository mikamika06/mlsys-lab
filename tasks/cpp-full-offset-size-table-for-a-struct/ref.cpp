#include "sol.hpp"

static int size_of(FieldType t) {
    switch (t) {
        case FieldType::Bool: return 1;
        case FieldType::Char: return 1;
        case FieldType::Short: return 2;
        case FieldType::Int: return 4;
        case FieldType::Long: return 8;
        case FieldType::LongLong: return 8;
        case FieldType::Float: return 4;
        case FieldType::Double: return 8;
        case FieldType::Pointer: return 8;
    }
    return 0;
}

void struct_layout(const FieldType* fields, int n, FieldLayout* out, int* total_size_out) {
    int off = 0;
    int max_align = 1;
    for (int i = 0; i < n; i++) {
        int size = size_of(fields[i]);
        int align = size; // natural alignment == size for every type here
        int pad = (align - (off % align)) % align;
        off += pad;
        out[i].offset = off;
        out[i].size = size;
        off += size;
        if (align > max_align) max_align = align;
    }
    int tail_pad = (max_align - (off % max_align)) % max_align;
    *total_size_out = off + tail_pad;
}
