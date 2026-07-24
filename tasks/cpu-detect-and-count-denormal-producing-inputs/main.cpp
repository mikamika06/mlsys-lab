#include <cfloat>
#include <cstdio>
#include <limits>
#include "sol.hpp"

// FIXED driver: 20 floats spanning ordinary normals, +-0, subnormals deep
// in range, subnormals right at the FLT_MIN/2 boundary, +-FLT_MIN itself
// (normal, NOT subnormal), +-infinity, and NaN.

int main() {
    const float MIN_NORM = FLT_MIN;  // smallest positive normal float
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    float arr[20] = {
        1.0f,
        -2.5f,
        100000.0f,
        0.0f,
        -0.0f,
        1e-40f,               // subnormal
        -1e-41f,              // subnormal
        1e-44f,               // subnormal (near FLT_TRUE_MIN)
        -1e-42f,              // subnormal
        MIN_NORM,              // normal (exact boundary, NOT subnormal)
        -MIN_NORM,              // normal (exact boundary, NOT subnormal)
        MIN_NORM * 0.5f,       // subnormal (exactly half of smallest normal)
        -(MIN_NORM * 0.5f),    // subnormal
        inf,
        -inf,
        nan,
        3.14159265f,
        -0.00001f,
        MIN_NORM * 1.5f,       // normal (just above the boundary)
        -(MIN_NORM * 1.5f),    // normal (just above the boundary)
    };

    int n = count_denormals(arr, 20);
    printf("%d\n", n);
    return 0;
}
