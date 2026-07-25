#include "sol.hpp"

double steady_state_mispredict_rate(double p) {
    // your code here
    double q = 1.0 - p;
    return p < q ? p : q;
}
