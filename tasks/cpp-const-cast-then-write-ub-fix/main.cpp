#include <cstdio>
#include "sol.hpp"

static void print_config(const char* label, const Config& c) {
    printf("%s version=%c threshold=%.3f flags=%d\n", label, c.version, c.threshold, c.flags);
}

int main() {
    const int NF = 3;
    int fixtures[NF] = {7, 42, -13};

    for (int i = 0; i < NF; i++) {
        Config c = make_config_with_flags(fixtures[i]);
        print_config("result", c);
        print_config("g_cfg ", g_cfg);   // g_cfg itself must never change
    }
    return 0;
}
