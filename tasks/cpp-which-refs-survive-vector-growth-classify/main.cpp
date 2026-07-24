#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver: 12 deterministic scenarios exercising std::vector reference
// invalidation. Prints one bit per scenario (1 = reference survives) plus the
// popcount. No randomness.
int main() {
    struct Scn { int n0, cap0, refIdx; std::vector<Op> ops; };
    std::vector<Scn> scns = {
        {4, 8, 1, {{PUSH_BACK, 0}}},                             //  1: room to grow -> survives
        {4, 4, 1, {{PUSH_BACK, 0}}},                             //  2: size==cap, realloc -> dies
        {4, 8, 2, {{RESERVE, 6}}},                               //  3: reserve <= cap, no-op -> survives
        {4, 8, 2, {{RESERVE, 20}}},                              //  4: reserve > cap, realloc -> dies
        {5, 8, 4, {{POP_BACK, 0}}},                              //  5: pop removes ref element -> dies
        {5, 8, 1, {{POP_BACK, 0}}},                              //  6: pop removes a later element -> survives
        {5, 8, 3, {{INSERT, 1}}},                                //  7: insert before ref, shifts it -> dies
        {5, 8, 1, {{INSERT, 3}}},                                //  8: insert after ref, no realloc -> survives
        {3, 3, 0, {{INSERT, 1}}},                                //  9: insert forces realloc -> dies
        {4, 8, 2, {{CLEAR, 0}}},                                 // 10: clear destroys elements -> dies
        {2, 8, 0, {{PUSH_BACK, 0}, {PUSH_BACK, 0}, {PUSH_BACK, 0}}}, // 11: 3 pushes stay in cap -> survives
        {3, 4, 0, {{PUSH_BACK, 0}, {PUSH_BACK, 0}}},             // 12: 2nd push crosses cap -> dies
    };

    int pop = 0;
    for (size_t i = 0; i < scns.size(); i++) {
        const Scn& s = scns[i];
        int bit = ref_survives(s.n0, s.cap0, s.refIdx, s.ops) ? 1 : 0;
        pop += bit;
        printf("%d ", bit);
    }
    printf("\npopcount=%d\n", pop);
    return 0;
}
