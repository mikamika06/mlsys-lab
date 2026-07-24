#include "sol.hpp"

double sum_value_compact(const Compact* arr, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        touch(&arr[i]);
        s += arr[i].value;
    }
    return s;
}

double sum_value_padded(const Padded* arr, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        touch(&arr[i]);
        s += arr[i].value;
    }
    return s;
}
