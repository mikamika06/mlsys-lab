#include <cstdio>
#include "sol.hpp"

// FIXED driver. Calls classify<T> for six different T, mixing integral and
// floating-point types, and prints the tag chosen for each.
int main() {
    int tags[6] = {
        classify<int>(5),
        classify<long>(5L),
        classify<char>('a'),
        classify<unsigned>(5u),
        classify<float>(1.5f),
        classify<double>(1.5),
    };
    const char* names[6] = {"int", "long", "char", "unsigned", "float", "double"};
    for (int i = 0; i < 6; i++) printf("%s=%d\n", names[i], tags[i]);
    return 0;
}
