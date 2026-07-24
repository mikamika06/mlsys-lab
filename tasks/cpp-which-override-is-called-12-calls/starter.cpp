#include "sol.hpp"

// TODO: For each of the 12 call sites listed in task.md (in order), decide which
// override actually runs and store its tag in out[i].
//
// Reason about, for every call:
//   * virtual vs non-virtual member,
//   * the STATIC type of the object/pointer/reference vs its DYNAMIC type,
//   * object slicing when a Derived is copied into a Base value.
//
// Tags: Base::who=1, Derived::who=2, MoreDerived::who=3,
//       Base::nonvirt=10, Derived::nonvirt=20.
void predict_tags(int out[12]) {
    for (int i = 0; i < 12; i++) out[i] = 0;  // placeholder — replace with your predictions
}
