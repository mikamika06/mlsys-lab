#include "sol.hpp"

// Reference: the correct alive/dead classification for each of the 24
// documented snippets in task.md, plus sizeof(Gadget) under LP64
// (`struct Gadget { int id; void* buffer; }` -> 4-byte int padded to 8-byte
// alignment for the pointer, + 8-byte pointer = 16 bytes).
int predict_lifetimes(bool out[24]) {
    bool bits[24] = {
        true,  false, false, true,  true,  true,  false, false,
        true,  true,  false, true,  false, true,  false, false,
        true,  true,  false, true,  false, false, true,  true,
    };
    for (int i = 0; i < 24; i++) out[i] = bits[i];
    return 16;
}
