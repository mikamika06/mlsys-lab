#include "sol.hpp"

// TODO: implement using two SFINAE/enable_if-constrained overloads (one
// viable only when T is integral, one only when T is floating-point) so
// the COMPILER picks the right specialization for each T. See sol.hpp.
template <typename T>
int classify(T x) {
    (void)x;
    // your code here
    return 0;
}

template int classify<int>(int);
template int classify<long>(long);
template int classify<char>(char);
template int classify<unsigned>(unsigned);
template int classify<float>(float);
template int classify<double>(double);
