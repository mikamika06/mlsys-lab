#include "sol.hpp"

double steady_state_mispredict_rate(double p) {
    double q = 1.0 - p;
    return (p * q) / (p * p + q * q);
}
