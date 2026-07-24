#include "sol.hpp"

// Fixed: __builtin_add_overflow is well-defined (never UB) regardless of
// optimization level, so both entry points agree on every input.
__attribute__((optnone)) bool check_no_overflow_noopt(int x) {
    int y;
    return !__builtin_add_overflow(x, 1, &y);
}

bool check_no_overflow_opt(int x) {
    int y;
    return !__builtin_add_overflow(x, 1, &y);
}
