#include "sol.hpp"
#include <cstddef>
#include <type_traits>

template <typename T>
struct MyDecay {
    using type = typename std::remove_cv<T>::type;
};

template <typename T>
struct MyDecay<T&> { using type = typename MyDecay<T>::type; };

template <typename T>
struct MyDecay<T&&> { using type = typename MyDecay<T>::type; };

template <typename T>
struct MyDecay<T[]> { using type = T*; };

template <typename T, std::size_t N>
struct MyDecay<T[N]> { using type = T*; };

template <typename Ret, typename... Args>
struct MyDecay<Ret(Args...)> { using type = Ret (*)(Args...); };

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
