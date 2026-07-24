#include "sol.hpp"

template <typename T, typename = void>
struct has_serialize : std::false_type {};

template <typename T>
struct has_serialize<T, std::void_t<
    decltype(std::declval<T>().serialize(std::declval<int>()))
>> : std::true_type {};

bool detect_DProbe1()  { return has_serialize<DProbe1>::value; }
bool detect_DProbe2()  { return has_serialize<DProbe2>::value; }
bool detect_DProbe3()  { return has_serialize<DProbe3>::value; }
bool detect_DProbe4()  { return has_serialize<DProbe4>::value; }
bool detect_DProbe5()  { return has_serialize<DProbe5>::value; }
bool detect_DProbe6()  { return has_serialize<DProbe6>::value; }
bool detect_DProbe7()  { return has_serialize<DProbe7>::value; }
bool detect_DProbe8()  { return has_serialize<DProbe8>::value; }
bool detect_DProbe9()  { return has_serialize<DProbe9>::value; }
bool detect_DProbe10() { return has_serialize<DProbe10>::value; }
bool detect_DProbe11() { return has_serialize<DProbe11>::value; }
bool detect_DProbe12() { return has_serialize<DProbe12>::value; }
