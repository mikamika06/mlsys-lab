#include "sol.hpp"

// TODO: implement the four ordered validation checks from sol.hpp, then
// return the sum of arr.data[0 .. arr.size). Right now this always returns
// 0 without validating anything, so every fixture that should raise an
// exception instead "succeeds" with the wrong value.
double validate_buffer(const BufferObj& arr, const std::string& expected_dtype,
                        const int* expected_shape, int expected_ndim) {
    (void)arr; (void)expected_dtype; (void)expected_shape; (void)expected_ndim;
    return 0.0;  // your code here
}
