#include <cstdio>
#include <atomic>
#include "sol.hpp"

// FIXED driver. Do not edit. Six fixed (write_mo, read_mo, data_val)
// cases, calls the learner's enumerator, and prints every outcome pair.
int main() {
    struct Case { std::memory_order w, r; int val; const char* label; };
    Case cases[] = {
        {std::memory_order_release, std::memory_order_acquire, 42,  "release/acquire"},
        {std::memory_order_relaxed, std::memory_order_acquire, 42,  "relaxed/acquire"},
        {std::memory_order_release, std::memory_order_relaxed, 100, "release/relaxed"},
        {std::memory_order_seq_cst, std::memory_order_seq_cst, 77,  "seq_cst/seq_cst"},
        {std::memory_order_acq_rel, std::memory_order_acquire, 15,  "acq_rel/acquire"},
        {std::memory_order_relaxed, std::memory_order_relaxed, 99,  "relaxed/relaxed"},
    };

    for (const auto& c : cases) {
        auto out = get_allowed_litmus_outcomes(c.w, c.r, c.val);
        printf("%s val=%d :", c.label, c.val);
        for (const auto& p : out) printf(" (%d,%d)", p.first, p.second);
        printf("\n");
    }
    return 0;
}
