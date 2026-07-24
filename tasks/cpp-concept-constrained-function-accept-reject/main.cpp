#include "sol.hpp"
#include <cstdio>

int main() {
    int out[12];
    classify_accepts(out);
    for (int i = 0; i < 12; i++) {
        printf("%d\n", out[i]);
    }
    return 0;
}
