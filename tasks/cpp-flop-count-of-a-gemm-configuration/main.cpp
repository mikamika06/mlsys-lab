#include <cstdio>
#include "sol.hpp"

int main() {
    printf("sizeof_GemmConfig %d\n", (int)sizeof(GemmConfig));

    GemmConfig cases[] = {
        {100, 200, 50, 1.0, 0.0},
        {50, 50, 10, 1.5, 1.0},
        {1, 1, 1000, 2.0, 0.0},
        {10, 10, 10, 0.5, -1.0},
        {7, 3, 1, 1.0, 0.0},
    };
    for (const auto& cfg : cases) {
        printf("flops %lld\n", gemmFlops(cfg));
    }
    return 0;
}
