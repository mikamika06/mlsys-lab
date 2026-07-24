#include "sol.hpp"

volatile int g_seed = 1;

static int compute_value() { return g_seed * 42; }

// BUG: a plain namespace-scope global. Its own dynamic initializer only
// runs when THIS translation unit's static initializers run — which, for
// this build, is AFTER main.cpp's (main.cpp is linked first). Reading it
// from main.cpp's earlier initializer sees it still zero-initialized.
int g_bad_value = compute_value();

int get_b_value() { return g_bad_value; }
