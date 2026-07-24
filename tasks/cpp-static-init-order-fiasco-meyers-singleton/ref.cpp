#include "sol.hpp"

// `volatile` so the compiler cannot constant-fold compute_value() away —
// it genuinely has to run at some point, not just at compile time.
volatile int g_seed = 1;

static int compute_value() { return g_seed * 42; }

// Meyers singleton: `value` is constructed the first time ANY caller, from
// ANY translation unit, passes through this line — including main.cpp's
// own dynamic initializer running before the rest of THIS TU has
// initialized. Safe regardless of cross-TU init order.
int get_b_value() {
    static int value = compute_value();
    return value;
}
