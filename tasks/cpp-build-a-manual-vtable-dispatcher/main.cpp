#include "sol.hpp"
#include <cstdio>

int main() {
    Derived1 d1(3, 1.5);
    Derived2 d2(7);
    Derived1 d1b(-2, 0.25);
    Derived2 d2b(-100);

    const Base* objs[4] = {&d1, &d2, &d1b, &d2b};
    const int xs[4] = {0, 5, -3, 42};

    for (int i = 0; i < 4; i++) {
        double r = manual_dispatch(objs[i], xs[i]);
        printf("%.6f\n", r);
    }
    return 0;
}
