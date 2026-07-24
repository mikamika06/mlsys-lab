#pragma once
#include <memory>
#include <vector>

// A subobject that lives *inside* a Parent. We never allocate a Child on its
// own heap block; it only exists as a member of some Parent.
struct Child {
    int value;
};

// The Parent owns a Child as a member subobject. When a Parent is destroyed it
// appends its id to the global destruction log g_dtor_log (defined in main.cpp),
// so a driver can observe destruction *order*.
struct Parent {
    int   id;
    Child child;
    ~Parent();          // definition below records the destruction
};

// Global destruction log: each entry is the id of a Parent whose destructor
// ran, in the order the destructors fired. Defined in main.cpp.
extern std::vector<int> g_dtor_log;

inline Parent::~Parent() { g_dtor_log.push_back(id); }

// CONTRACT ------------------------------------------------------------------
// Return a std::shared_ptr<Child> that *points to* parent->child (a subobject
// of the Parent) while *sharing ownership* with `parent`. The returned pointer
// must keep the whole Parent alive: as long as it (or any copy of it) is alive,
// the Parent must NOT be destroyed, even after every shared_ptr<Parent> owner
// has been dropped. When the last such alias dies, the Parent destructor runs.
//
// Use the shared_ptr *aliasing constructor*.
std::shared_ptr<Child> alias_child(const std::shared_ptr<Parent>& parent);
