#pragma once

// checkType_i() must return true iff YOUR OWN template<typename T> struct
// MyDecay, implemented from scratch in this file, produces EXACTLY the same
// type (checked with std::is_same) as the real std::decay_t<T>, for the
// concrete type T_i listed next to each function below and defined in this
// same .cpp file (templates need their definition visible at the point of
// instantiation, so MyDecay and all of its specializations live entirely in
// THIS file, not in this shared header).
//
// std::decay<T>::type is:
//   1. If T is a reference (T& or T&&): decay(remove_reference_t<T>).
//   2. Else if T is an array type Elem[N] (or Elem[]): Elem*.
//   3. Else if T is a function type Ret(Args...): Ret(*)(Args...).
//   4. Else: remove top-level cv-qualifiers only (remove_cv_t<T>) -- a
//      pointee's own qualifiers are NOT top-level and must survive, e.g.
//      `const int* const` decays to `const int*`, not `int*`.
//
//   T1  = int                    T6  = const volatile short   T11 = int(double)
//   T2  = const int              T7  = int[10]                T12 = char*
//   T3  = const int&             T8  = const char[5]          T13 = double[3]
//   T4  = int&&                  T9  = const int* const       T14 = const float
//   T5  = volatile double&       T10 = void()                 T15 = unsigned int&
bool checkType1();
bool checkType2();
bool checkType3();
bool checkType4();
bool checkType5();
bool checkType6();
bool checkType7();
bool checkType8();
bool checkType9();
bool checkType10();
bool checkType11();
bool checkType12();
bool checkType13();
bool checkType14();
bool checkType15();
