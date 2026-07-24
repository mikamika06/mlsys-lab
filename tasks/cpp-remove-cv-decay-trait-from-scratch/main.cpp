#include <cstdio>
#include "sol.hpp"

int main() {
    bool (*checks[15])() = {
        checkType1, checkType2, checkType3, checkType4, checkType5,
        checkType6, checkType7, checkType8, checkType9, checkType10,
        checkType11, checkType12, checkType13, checkType14, checkType15,
    };
    for (int i = 0; i < 15; i++) {
        printf("type%d %d\n", i + 1, checks[i]() ? 1 : 0);
    }
    return 0;
}
