#include "sol.hpp"

// BUG: no anonymous namespace. `Config` and `get_size` have ordinary
// external linkage; `get_size` is marked `inline` so the duplicate
// definition doesn't hard-fail the link, but that makes it a weak symbol
// with the SAME mangled name as main.cpp's `get_size`. The real linker
// merges them -- main.cpp is first on the command line, so its definition
// wins for the whole program, and the call below silently reports
// main.cpp's Config size instead of this file's own.
struct Config { int x; double y; };
inline __attribute__((noinline)) int get_size() { return (int)sizeof(Config); }

int reportSize() { return get_size(); }
