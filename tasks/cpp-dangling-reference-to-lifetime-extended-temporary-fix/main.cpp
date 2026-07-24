#include <cstdio>
#include "sol.hpp"

// FIXED driver. Three (id, val) cases, matching the three-case coverage of
// the original grader. Each result is read back and printed.

int main() {
    int ids[3] = {1, 2, 42};
    float vals[3] = {3.14f, -2.5f, 10.0f};

    for (int i = 0; i < 3; ++i) {
        Result r = get_result(ids[i], vals[i]);
        printf("%d %.6f\n", r.id, r.val);
    }
    return 0;
}
