#pragma once

// classify<T>(x) must return 0 if T is an integral type, 1 if T is a
// floating-point type -- decided via SFINAE / std::enable_if-constrained
// overload resolution AT COMPILE TIME (two templates, each viable for only
// one category of T, so the compiler picks one and the other is simply
// never instantiated) -- NOT a runtime std::is_integral<T>::value branch
// inside one generic body.
//
// Defined (with explicit instantiations for the types main.cpp calls) in
// ref.cpp / solve.cpp, not here -- only the declaration lives in this
// header.
template <typename T>
int classify(T x);
