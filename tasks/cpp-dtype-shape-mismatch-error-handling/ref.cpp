#include "sol.hpp"

double validate_buffer(const BufferObj& arr, const std::string& expected_dtype,
                        const int* expected_shape, int expected_ndim) {
    if (!arr.is_valid_buffer)
        throw TypeErrorSim("Input must be a numpy.ndarray buffer");

    if (arr.dtype != expected_dtype)
        throw TypeErrorSim("Dtype mismatch");

    if (arr.ndim != expected_ndim)
        throw ValueErrorSim("Dimension mismatch");

    for (int i = 0; i < arr.ndim; i++) {
        if (expected_shape[i] != -1 && arr.shape[i] != expected_shape[i])
            throw ValueErrorSim("Shape mismatch");
    }

    double sum = 0.0;
    for (int i = 0; i < arr.size; i++) sum += arr.data[i];
    return sum;
}
