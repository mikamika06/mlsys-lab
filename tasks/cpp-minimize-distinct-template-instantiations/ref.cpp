#include "sol.hpp"

void processAll() {
    int i = 42;
    float f = 3.5f;
    double d = 9.25;

    // all three numeric values funneled through ONE shared instantiation
    process<double>((double)i);
    process<double>((double)f);
    process<double>(d);

    process<const char*>("alpha");
    process<const char*>("beta");
}
