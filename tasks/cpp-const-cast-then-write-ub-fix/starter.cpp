#include "sol.hpp"

// BUG: g_cfg is genuinely `const` (see sol.hpp). const_cast-ing it away and
// writing is Undefined Behavior — at -O2 the compiler assumes g_cfg never
// changes, drops the write, and keeps folding every read of g_cfg back to
// its original initializer. `flags` silently stays 0 no matter what
// new_flags is. Fix this by copying g_cfg into a real mutable local instead.
Config make_config_with_flags(int new_flags) {
    const_cast<Config&>(g_cfg).flags = new_flags;
    return g_cfg;
}
