#include "sol.hpp"

#include <cstdio>
#include <string>
#include <type_traits>

// TODO: classify a real type T using compile-time type traits
// (std::is_lvalue_reference_v, std::is_rvalue_reference_v, std::is_pointer_v,
// std::is_const_v, std::remove_reference_t, std::remove_pointer_t) -- peel
// off one layer at a time and recurse on what's left. See task.md for the
// label format.
template <typename T>
std::string classify_type() {
    (void)0;
    // your code here
    return "";
}

void run_declaration_tests() {
    printf("1 %s\n", classify_type<int*>().c_str());
    printf("2 %s\n", classify_type<const int*>().c_str());
    printf("3 %s\n", classify_type<int* const>().c_str());
    printf("4 %s\n", classify_type<const int* const>().c_str());
    printf("5 %s\n", classify_type<int&>().c_str());
    printf("6 %s\n", classify_type<const int&>().c_str());
    printf("7 %s\n", classify_type<int&&>().c_str());
    printf("8 %s\n", classify_type<const int&&>().c_str());
    printf("9 %s\n", classify_type<int**>().c_str());
    printf("10 %s\n", classify_type<const int**>().c_str());
    printf("11 %s\n", classify_type<int* const*>().c_str());
    printf("12 %s\n", classify_type<int*&>().c_str());
}
