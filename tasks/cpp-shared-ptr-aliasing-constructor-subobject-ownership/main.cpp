#include <cstdio>
#include <memory>
#include <vector>
#include "sol.hpp"

// The one and only definition of the global destruction log.
std::vector<int> g_dtor_log;

int main() {
    // Build a Parent with id=7 and child.value=42 on a shared control block.
    std::shared_ptr<Parent> p = std::make_shared<Parent>();
    p->id          = 7;
    p->child.value = 42;

    // Hand out an alias to the *subobject* that co-owns the Parent.
    std::shared_ptr<Child> a = alias_child(p);

    // While both p and the alias are alive, ownership is shared: count == 2.
    int uc_both = (int)a.use_count();

    // Value read through the alias while the Parent is alive.
    int v1 = a ? a->value : -1;

    // Drop the ONLY shared_ptr<Parent> owner. A correct alias keeps the
    // Parent alive, so no destructor should have fired yet.
    p.reset();
    int dtors_after_parent_reset = (int)g_dtor_log.size();   // correct: 0

    // The alias is now the sole owner: count == 1.
    int uc_alias = (int)a.use_count();                       // correct: 1

    // The subobject is still valid and readable through the alias.
    int v2 = a ? a->value : -1;                              // correct: 42

    // Drop the last alias -> the Parent finally gets destroyed now.
    a.reset();
    int dtors_after_alias_reset = (int)g_dtor_log.size();    // correct: 1

    // Which Parent id was destroyed (proves the right object ran its dtor).
    int destroyed_id = g_dtor_log.empty() ? -1 : g_dtor_log.back(); // correct: 7

    printf("%d\n", uc_both);
    printf("%d\n", v1);
    printf("%d\n", dtors_after_parent_reset);
    printf("%d\n", uc_alias);
    printf("%d\n", v2);
    printf("%d\n", dtors_after_alias_reset);
    printf("%d\n", destroyed_id);
    return 0;
}
