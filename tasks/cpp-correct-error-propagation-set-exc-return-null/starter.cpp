#include "sol.hpp"

// TODO: implement the error-propagation protocol described in sol.hpp.
// Right now this always returns nullptr WITHOUT setting g_exc, and never
// computes a result for valid input — both wrong.
PyFloatObj* safe_divide(double a, double b) {
    (void)a; (void)b;
    return nullptr;  // your code here
}
