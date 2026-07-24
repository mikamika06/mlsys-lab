#include "sol.hpp"

#include <cstdio>
#include <type_traits>
#include <utility>

static int g_tag = 0;

// Reference: mutually exclusive constraints.
//   Overload 1 (tag 1): integral types with sizeof(T) > 4
//   Overload 2 (tag 2): ANY type with sizeof(T) <= 4
//   Overload 3 (tag 3): floating-point types with sizeof(T) > 4
// Every scalar type satisfies at most one of these, so overload resolution
// is never ambiguous.
template <typename T, std::enable_if_t<std::is_integral_v<T> && (sizeof(T) > 4), int> = 0>
void process(T) { g_tag = 1; }

template <typename T, std::enable_if_t<(sizeof(T) <= 4), int> = 0>
void process(T) { g_tag = 2; }

template <typename T, std::enable_if_t<std::is_floating_point_v<T> && (sizeof(T) > 4), int> = 0>
void process(T) { g_tag = 3; }

// Harness-side detection idiom: does `process(T{})` compile at all for this
// T? A genuinely ambiguous or genuinely no-viable-candidate call is invalid
// in the immediate context of this decltype, so this safely evaluates to
// false in either case instead of taking the whole program down with it.
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
