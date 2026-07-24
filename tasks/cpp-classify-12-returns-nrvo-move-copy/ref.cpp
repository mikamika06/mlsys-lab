#include "sol.hpp"

// Reference: the correct classification for f1..f12, and sizeof(T) under
// LP64 (`char c; double d; int i;` -> the double forces 8-byte alignment,
// so `c` is padded to 8, `d` is 8, `i` is 4 padded to 8 -> 24 bytes total).
void predict_return_kinds(std::string out[12]) {
    static const char* labels[12] = {
        "nrvo", "rvo",  "move", "copy", "move", "copy",
        "copy", "copy", "copy", "nrvo", "copy", "rvo",
    };
    for (int i = 0; i < 12; i++) out[i] = labels[i];
}

int predict_struct_size() { return 24; }
