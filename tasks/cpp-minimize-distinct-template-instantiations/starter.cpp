#include "sol.hpp"

// BUG: each numeric type gets its own template instantiation instead of
// being funneled through a single shared one -- 4 distinct symbols
// (int, float, double, const char*) instead of 2.
void processAll() {
    int i = 42;
    float f = 3.5f;
    double d = 9.25;

    process<int>(i);
    process<float>(f);
    process<double>(d);

    process<const char*>("alpha");
    process<const char*>("beta");
}
