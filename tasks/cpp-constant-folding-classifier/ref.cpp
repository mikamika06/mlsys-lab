#include "sol.hpp"
#include <cstdlib>
#include <cmath>
#include <ctime>

namespace {

// Classic "is this expression usable in a constant expression" probe: a
// lambda wrapping the expression is invoked as the initializer of a
// non-type template parameter, a context that requires a converted
// constant expression. If the compiler can prove the call is a constant
// expression, the primary template is selected (folds -> true). If not,
// substitution fails (SFINAE) and the ellipsis fallback is selected
// (does not fold -> false) -- the lambda body is never actually run in
// that case.
template <typename Lambda, int = (Lambda{}(), 0)>
constexpr bool is_constexpr_impl(Lambda) { return true; }
constexpr bool is_constexpr_impl(...) { return false; }

}  // namespace

#define IS_FOLDABLE(...) is_constexpr_impl([]{ return (__VA_ARGS__); })

std::vector<int> classify_constant_folding() {
    std::vector<int> r;
    r.push_back(IS_FOLDABLE(2 + 3 * 4) ? 1 : 0);
    r.push_back(IS_FOLDABLE(sizeof(int)) ? 1 : 0);
    r.push_back(IS_FOLDABLE(sizeof(int) * sizeof(double)) ? 1 : 0);
    r.push_back(IS_FOLDABLE(1 << 8) ? 1 : 0);
    r.push_back(IS_FOLDABLE(0xFF & 0x0F) ? 1 : 0);
    r.push_back(IS_FOLDABLE(__builtin_popcount(0xABCDu)) ? 1 : 0);
    r.push_back(IS_FOLDABLE(([]{ int arr[10]; return sizeof(arr); })()) ? 1 : 0);
    r.push_back(IS_FOLDABLE(([]{ constexpr int k = 7; return k * 2; })()) ? 1 : 0);
    r.push_back(IS_FOLDABLE(1 / 0) ? 1 : 0);
    r.push_back(IS_FOLDABLE(rand()) ? 1 : 0);
    r.push_back(IS_FOLDABLE(std::sin(1.0)) ? 1 : 0);
    r.push_back(IS_FOLDABLE(std::time(nullptr)) ? 1 : 0);
    return r;
}
