#include "sol.hpp"

// TODO: return a shared_ptr<Child> that points to &parent->child but shares
// ownership with `parent` (use the shared_ptr aliasing constructor), so the
// Parent stays alive until the last returned alias is dropped.
std::shared_ptr<Child> alias_child(const std::shared_ptr<Parent>& parent) {
    // your code here
    return {};   // wrong: a null pointer owns nothing and keeps nothing alive
}
