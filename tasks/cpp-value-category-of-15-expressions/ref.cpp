#include "sol.hpp"
#include <string>
#include <type_traits>
#include <utility>

namespace {

template <typename T>
std::string category_of() {
    if (std::is_lvalue_reference<T>::value) return "lvalue";
    if (std::is_rvalue_reference<T>::value) return "xvalue";
    return "prvalue";
}

}  // namespace

// decltype((expr)) (double parens) reports the real value category of
// `expr` as its own reference-qualification: T& for lvalue, T&& for
// xvalue, plain T for prvalue.
#define VCAT(expr) category_of<decltype((expr))>()

std::vector<std::string> classify_value_categories() {
    int x = 1;
    std::vector<std::string> r;
    r.push_back(VCAT(x));
    r.push_back(VCAT(42));
    r.push_back(VCAT(std::move(x)));
    r.push_back(VCAT("hello"));
    r.push_back(VCAT(std::string("tmp")));
    r.push_back(VCAT(*(&x)));
    r.push_back(VCAT(x + 1));
    r.push_back(VCAT(++x));
    r.push_back(VCAT(x++));
    r.push_back(VCAT(static_cast<int&&>(x)));
    r.push_back(VCAT(std::declval<int&>()));
    r.push_back(VCAT(std::declval<int&&>()));
    r.push_back(VCAT(std::string("a") + std::string("b")));
    r.push_back(VCAT(std::move(*(&x))));
    r.push_back(VCAT((x = 1)));
    return r;
}
