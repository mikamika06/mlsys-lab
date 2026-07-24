#include "sol.hpp"

PyObj* array_sum_ext(PyObj* buffer) {
    double sum = 0.0;
    for (int i = 0; i < buffer->n; ++i) sum += buffer->data[i];

    double* payload = new double[1]{sum};
    return new PyObj{payload, 1, 1};
}
