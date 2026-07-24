#include "sol.hpp"

// TODO: classify whether the compiler may assume type_a* and type_b* do not
// alias. See sol.hpp for the full rule set (cv/sign stripping, char/byte
// exception, identical-type exception, base/derived exception).
int may_assume_no_alias(const std::string& type_a, const std::string& type_b,
                         const std::vector<std::pair<std::string, std::string>>& hierarchy) {
    (void)type_a;
    (void)type_b;
    (void)hierarchy;
    return 1;
}
