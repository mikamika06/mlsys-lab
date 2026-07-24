#include "sol.hpp"

void struct_layout(const FieldType* fields, int n, FieldLayout* out, int* total_size_out) {
    // your code here
    for (int i = 0; i < n; i++) {
        out[i].offset = 0;
        out[i].size = 0;
    }
    *total_size_out = 0;
}
