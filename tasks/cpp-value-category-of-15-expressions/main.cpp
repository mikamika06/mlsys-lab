#include <cstdio>
#include "sol.hpp"

// FIXED driver. Do not edit. Calls the learner's classifier and prints
// the 15 labels, one per line.
int main() {
    auto labels = classify_value_categories();
    for (const auto& s : labels) printf("%s\n", s.c_str());
    return 0;
}
