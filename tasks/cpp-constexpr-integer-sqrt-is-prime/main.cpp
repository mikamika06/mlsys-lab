#include "sol.hpp"
#include <cstdio>

int main() {
    static const int values[] = {
        0, 1, 2, 3, 4, 16, 17, 49,
        10, 11, 25, 29, 97, 100,
        31, 32, 33, 101, 127, 128,
        997, 1000, 1024, 1031,
    };
    int n = static_cast<int>(sizeof(values) / sizeof(values[0]));
    for (int i = 0; i < n; i++) {
        int v = values[i];
        printf("%d %d %d\n", integer_sqrt(v), is_prime(v) ? 1 : 0, v);
    }
    return 0;
}
