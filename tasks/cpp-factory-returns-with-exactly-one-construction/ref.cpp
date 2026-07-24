#include "sol.hpp"

// Return the freshly built Probe directly as a prvalue: C++17 guaranteed
// copy elision constructs it in place at the caller's slot. Exactly one
// constructor call (the direct one) happens in total.
Probe make_probe(int tag, double payload) {
    return Probe(tag, payload);
}
