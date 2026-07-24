#include "sol.hpp"
#include <cmath>

float fma_result(float a, float b, float c) {
    return std::fma(a, b, c);
}

float naive_result(float a, float b, float c) {
    volatile float p = a * b;
    return p + c;
}
