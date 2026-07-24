#include <cstdio>
#include <vector>
#include "sol.hpp"

// PROVIDED. Five real element types with real, compiler-computed sizes
// (natural alignment and tail padding included) -- the elem_size passed to
// deque_mapping is never guessed, it comes straight from `sizeof`.
struct CaseA { char a; int b; };
struct CaseB { double a; void* b; double c; };
struct CaseC { char a; };
struct CaseD { char a; double b; char c; char d; };
using CaseE = double[100];

// FIXED driver. Do not edit. Five fixed (elem_size, first_offset, indices)
// cases, calls the learner's deque_mapping for each, and prints the
// resulting (block, offset) pairs.
int main() {
    struct Case { long elem_size; long first_offset; std::vector<long> indices; };
    std::vector<Case> cases = {
        {(long)sizeof(CaseA), 60,  {0, 3, 4, 100}},
        {(long)sizeof(CaseB), 0,   {0, 5, 20, 21}},
        {(long)sizeof(CaseC), 500, {0, 10, 20, 511, 512}},
        {(long)sizeof(CaseD), 2,   {0, 1, 2, 3}},
        {(long)sizeof(CaseE), 0,   {0, 1, 2, 3}},
    };

    for (const auto& c : cases) {
        auto result = deque_mapping(c.elem_size, c.first_offset, c.indices);
        printf("elem_size=%ld first_offset=%ld :", c.elem_size, c.first_offset);
        for (const auto& p : result) printf(" (%ld,%ld)", p.first, p.second);
        printf("\n");
    }
    return 0;
}
