#include "sol.hpp"

// TODO: implement both factories (see sol.hpp for the exact required pattern
// each one must use). Right now by-value returns an empty Matrix and the
// out-param is left untouched, so both the content and the copy counts fail.
Matrix make_by_value(int n) {
    (void)n;
    return Matrix();  // your code here
}

void make_out_param(int n, Matrix& out) {
    (void)n; (void)out;  // your code here
}
