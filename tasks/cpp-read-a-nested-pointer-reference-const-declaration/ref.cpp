#include "sol.hpp"

#include <cstdio>
#include <string>
#include <type_traits>

// Reference: classify a real type T using real compile-time type traits --
// no string parsing at all. Peel off one layer (reference, then pointer)
// at a time and recurse on what's left, exactly the way the C++ grammar
// itself is nested.
template <typename T>
std::string classify_type() {
    if constexpr (std::is_lvalue_reference_v<T>) {
        return "ref-to-" + classify_type<std::remove_reference_t<T>>();
    } else if constexpr (std::is_rvalue_reference_v<T>) {
        return "rvalue-ref-to-" + classify_type<std::remove_reference_t<T>>();
    } else if constexpr (std::is_pointer_v<T>) {
        std::string prefix = std::is_const_v<T> ? "const-pointer-to-" : "pointer-to-";
        return prefix + classify_type<std::remove_pointer_t<T>>();
    } else {
        return std::is_const_v<T> ? "const-int" : "int";
    }
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
