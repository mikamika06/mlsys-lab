#include <cstdio>
#include "sol.hpp"

static void run_case(long n) {
    double* arr = new double[n];
    for (long i = 0; i < n; i++) arr[i] = i * 1.5 + 1.0;

    ArrayView view = make_zero_copy_view(arr, n);

    printf("n=%ld pointers_equal=%d len_ok=%d itemsize_ok=%d stride_ok=%d\n",
           n,
           (view.buf == arr) ? 1 : 0,
           (view.len == n) ? 1 : 0,
           (view.itemsize == (long)sizeof(double)) ? 1 : 0,
           (view.stride == (long)sizeof(double)) ? 1 : 0);

    // Mutate through the view, then read directly from `arr` -- must see it.
    for (long i = 0; i < n; i++) view.buf[i] = view.buf[i] * 2.0 - 1.0;

    printf("arr-after-write:");
    for (long i = 0; i < n; i++) printf(" %.3f", arr[i]);
    printf("\n");

    delete[] arr;
}

int main() {
    run_case(3);
    run_case(1);
    run_case(6);
    return 0;
}
