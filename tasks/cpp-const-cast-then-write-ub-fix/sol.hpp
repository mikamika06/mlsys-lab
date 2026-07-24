#pragma once

struct Config {
    char version;
    double threshold;
    int flags;
};

// The ORIGINAL config, genuinely declared `const`, with its initializer
// visible right here so every translation unit that includes this header
// sees the same compile-time-known value. Casting away its constness with
// const_cast and then writing through it is Undefined Behavior: because the
// object is genuinely const, the compiler is free to assume it never
// changes, and at -O2 it exploits that by folding every read of g_cfg back
// to this initializer and dropping the write entirely — a real, observable
// miscompile, not a hypothetical one.
inline const Config g_cfg = {'A', 0.5, 0};

// Return a Config with the same version/threshold as g_cfg but flags set to
// new_flags. Do NOT mutate g_cfg (directly or via const_cast) — build a
// genuinely mutable local copy instead and write to that.
Config make_config_with_flags(int new_flags);
