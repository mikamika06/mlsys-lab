#include "sol.hpp"

// Aliasing constructor: shares the control block of `parent` (so the Parent's
// reference count is incremented and the Parent stays alive) while the stored
// pointer is &parent->child. No new allocation, no separate deleter.
std::shared_ptr<Child> alias_child(const std::shared_ptr<Parent>& parent) {
    return std::shared_ptr<Child>(parent, &parent->child);
}
