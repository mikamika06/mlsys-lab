#include <cstdio>
#include "sol.hpp"

// FIXED driver. Do not edit. Calls the learner's classifier and prints
// the 12 labels, one per line.
int main() {
    auto labels = smart_pointer_selection();
    for (const auto& s : labels) printf("%s\n", s.c_str());
    return 0;
}
