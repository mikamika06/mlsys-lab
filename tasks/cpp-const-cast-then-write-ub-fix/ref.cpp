#include "sol.hpp"

// Fixed: copy g_cfg into a genuinely mutable local, then write to the copy.
Config make_config_with_flags(int new_flags) {
    Config local = g_cfg;
    local.flags = new_flags;
    return local;
}
