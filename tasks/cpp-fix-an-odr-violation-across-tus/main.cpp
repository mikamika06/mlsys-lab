#include <cstdio>
#include "sol.hpp"

// This TU's own Config -- unrelated to (and smaller than) whatever Config
// the other file defines. Ordinary external linkage, on purpose: this
// stands in for "some inline function already living in a shared header".
struct Config { int x; };
inline __attribute__((noinline)) int get_size() { return (int)sizeof(Config); }

int main() {
    int mainSize = get_size();
    int otherSize = reportSize();
    printf("main_size %d\n", mainSize);
    printf("other_size %d\n", otherSize);
    return 0;
}
