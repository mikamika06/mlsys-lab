#include "sol.hpp"
#include <type_traits>

// TODO: implement MyDecay from scratch (reference removal, array-to-pointer
// decay, function-to-function-pointer decay, top-level cv stripping). This
// version is just the identity -- no specializations at all.
template <typename T>
struct MyDecay {
    using type = T;   // your code here
};

template <typename T>
static bool check() {
    return std::is_same<typename MyDecay<T>::type, typename std::decay<T>::type>::value;
}

bool checkType1()  { return check<int>(); }
bool checkType2()  { return check<const int>(); }
bool checkType3()  { return check<const int&>(); }
bool checkType4()  { return check<int&&>(); }
bool checkType5()  { return check<volatile double&>(); }
bool checkType6()  { return check<const volatile short>(); }
bool checkType7()  { return check<int[10]>(); }
bool checkType8()  { return check<const char[5]>(); }
bool checkType9()  { return check<const int* const>(); }
bool checkType10() { return check<void()>(); }
bool checkType11() { return check<int(double)>(); }
bool checkType12() { return check<char*>(); }
bool checkType13() { return check<double[3]>(); }
bool checkType14() { return check<const float>(); }
bool checkType15() { return check<unsigned int&>(); }
