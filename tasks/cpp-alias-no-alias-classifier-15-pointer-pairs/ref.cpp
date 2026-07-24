#include "sol.hpp"

namespace {

std::string strip_cv_sign(std::string t) {
    static const char* prefixes[] = {"const ", "volatile ", "unsigned ", "signed "};
    for (const char* p : prefixes) {
        std::string pre(p);
        size_t pos = t.find(pre);
        while (pos != std::string::npos) {
            t.erase(pos, pre.size());
            pos = t.find(pre);
        }
    }
    return t;
}

bool is_base(const std::string& base, const std::string& derived,
             const std::vector<std::pair<std::string, std::string>>& hierarchy) {
    std::string cur = derived;
    while (true) {
        if (cur == base) return true;
        std::string next;
        bool found = false;
        for (const auto& kv : hierarchy) {
            if (kv.first == cur) { next = kv.second; found = true; break; }
        }
        if (!found || next.empty()) return false;
        cur = next;
    }
}

}  // namespace

int may_assume_no_alias(const std::string& type_a, const std::string& type_b,
                         const std::vector<std::pair<std::string, std::string>>& hierarchy) {
    std::string t1 = strip_cv_sign(type_a);
    std::string t2 = strip_cv_sign(type_b);

    if (t1 == "char" || t1 == "std::byte" || t2 == "char" || t2 == "std::byte") return 0;
    if (t1 == t2) return 0;
    if (is_base(t1, t2, hierarchy) || is_base(t2, t1, hierarchy)) return 0;
    return 1;
}
