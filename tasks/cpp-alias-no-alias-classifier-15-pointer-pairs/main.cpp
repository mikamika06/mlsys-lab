#include <cstdio>
#include "sol.hpp"

// FIXED driver. Do not edit. Builds a fixed class hierarchy and 15 fixed
// pointer-type pairs, classifies each, and prints the 15 results as a
// space-separated 0/1 sequence.
int main() {
    std::vector<std::pair<std::string, std::string>> hierarchy = {
        {"Derived", "Base"},
        {"Derived2", "Base"},
        {"Base", ""},
        {"Unrelated", ""},
    };

    std::vector<std::pair<std::string, std::string>> pairs = {
        {"int", "int"},
        {"int", "float"},
        {"int", "unsigned int"},
        {"float", "char"},
        {"std::byte", "double"},
        {"Base", "Derived"},
        {"Derived", "Base"},
        {"Derived", "Derived2"},
        {"Base", "Unrelated"},
        {"const int", "int"},
        {"signed short", "unsigned short"},
        {"char", "unsigned char"},
        {"long", "long long"},
        {"Derived2", "Unrelated"},
        {"volatile float", "float"},
    };

    for (const auto& p : pairs) {
        int r = may_assume_no_alias(p.first, p.second, hierarchy);
        printf("%d ", r);
    }
    printf("\n");
    return 0;
}
