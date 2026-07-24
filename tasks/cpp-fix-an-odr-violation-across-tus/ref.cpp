#include "sol.hpp"

namespace {
    struct Config { int x; double y; };
    __attribute__((noinline)) int get_size() { return (int)sizeof(Config); }
}

int reportSize() { return get_size(); }
