#include "sol.hpp"
#include <type_traits>

// Two overloads, mutually exclusive via enable_if: for any T, exactly one
// is a viable candidate, so overload resolution -- not a runtime branch --
// picks the right one.
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
int classify_impl(T) { return 0; }

template <typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
int classify_impl(T) { return 1; }

template <typename T>
int classify(T x) { return classify_impl(x); }

// Explicit instantiations for every type main.cpp calls (required since
// the definition lives here, in a separate translation unit from main.cpp).
template int classify<int>(int);
template int classify<long>(long);
template int classify<char>(char);
template int classify<unsigned>(unsigned);
template int classify<float>(float);
template int classify<double>(double);
