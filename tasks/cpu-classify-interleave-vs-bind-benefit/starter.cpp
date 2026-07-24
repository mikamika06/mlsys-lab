#include "sol.hpp"

const char* classify_workload(const Workload& w, double* bind_ns, double* interleave_ns) {
    // your code here
    *bind_ns = 0.0;
    *interleave_ns = 0.0;
    return "bind";
}
