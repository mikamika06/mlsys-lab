#include <cstdio>
#include <string>
#include <vector>
#include "sol.hpp"

// FIXED driver + FIXED log_event(). Replays four op sequences against four
// independent MiniVariant instances, then prints the whole event log plus
// sizeof(MiniVariant) (the real compiler's own answer).

static std::vector<std::string> g_log;

void log_event(const char* s) { g_log.push_back(s); }

int main() {
    // Case 1: set A, get A, set B, get B, destroy.
    {
        MiniVariant v;
        variant_set_a(v);
        variant_get_a(v);
        variant_set_b(v);
        variant_get_b(v);
        variant_destroy(v);
    }

    // Case 2: set A, get B (wrong type -> invalid), set B, get B.
    {
        MiniVariant v;
        variant_set_a(v);
        variant_get_b(v);
        variant_set_b(v);
        variant_get_b(v);
        variant_destroy(v);   // trailing cleanup
    }

    // Case 3: set A, set A AGAIN (must still dtor+ctor, same type), get A,
    // set B, destroy.
    {
        MiniVariant v;
        variant_set_a(v);
        variant_set_a(v);
        variant_get_a(v);
        variant_set_b(v);
        variant_destroy(v);
    }

    // Case 4: get A before anything is set (-> invalid), set A, get A,
    // set B (leave B active — trailing cleanup below).
    {
        MiniVariant v;
        variant_get_a(v);
        variant_set_a(v);
        variant_get_a(v);
        variant_set_b(v);
        variant_destroy(v);   // trailing cleanup
    }

    for (const auto& e : g_log) {
        printf("%s\n", e.c_str());
    }
    printf("sizeof(MiniVariant)=%d\n", static_cast<int>(sizeof(MiniVariant)));
    return 0;
}
