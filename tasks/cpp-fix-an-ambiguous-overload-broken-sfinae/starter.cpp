#include "sol.hpp"

#include <cstdio>
#include <type_traits>
#include <utility>

static int g_tag = 0;

// BUG: these constraints overlap. Overload 1 accepts every integral type
// (no size check), Overload 2 accepts every type with sizeof(T) <= 4 -- so
// any small integral type (bool, char, short, int) satisfies BOTH, and
// Overload 3 accepts every floating-point type (no size check), which
// overlaps Overload 2 for `float`. Fix the constraints so each type
// satisfies at most one overload -- see task.md for the intended split.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void process(T) { g_tag = 1; }

template <typename T, std::enable_if_t<(sizeof(T) <= 4), int> = 0>
void process(T) { g_tag = 2; }

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
void process(T) { g_tag = 3; }

// Harness-side detection idiom: does `process(T{})` compile at all for this
// T? A genuinely ambiguous or genuinely no-viable-candidate call is invalid
// in the immediate context of this decltype, so this safely evaluates to
// false in either case instead of taking the whole program down with it.
// (Do not edit -- this is the same in both files, and is not the bug.)
template <typename T, typename = void>
struct can_process : std::false_type {};
template <typename T>
struct can_process<T, std::void_t<decltype(process(std::declval<T>()))>> : std::true_type {};

template <typename T>
static void test_type(const char* name) {
    if constexpr (can_process<T>::value) {
        g_tag = 0;
        process(T{});
        printf("%s %d\n", name, g_tag);
    } else {
        printf("%s NoMatch\n", name);
    }
}

void run_overload_tests() {
    test_type<bool>("bool");
    test_type<char>("char");
    test_type<short>("short");
    test_type<int>("int");
    test_type<long>("long");
    test_type<long long>("long long");
    test_type<float>("float");
    test_type<double>("double");
    test_type<int*>("pointer");
}
