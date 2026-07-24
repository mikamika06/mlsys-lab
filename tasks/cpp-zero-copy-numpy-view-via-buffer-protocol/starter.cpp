#include "sol.hpp"

// TODO: implement make_zero_copy_view per sol.hpp -- it must alias `arr`,
// not allocate anything. Right now it returns an all-zero/null view, which
// fails every check.
ArrayView make_zero_copy_view(double* arr, long n) {
    (void)arr; (void)n;
    ArrayView v{nullptr, 0, 0, 0};  // your code here
    return v;
}
