#include "sol.hpp"

PyFloatObj* safe_divide(double a, double b) {
    if (b == 0.0) {
        set_error(ExcType::ZeroDivisionError, "division by zero");
        return nullptr;
    }
    return new PyFloatObj{a / b};
}
