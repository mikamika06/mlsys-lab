#pragma once
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// For each of the 12 ownership scenarios below, return the single most
// appropriate pointer type, one of exactly "unique", "shared", "weak", or
// "raw", IN THIS ORDER:
//
//    1. A resource exclusively owned by a single class, automatically
//       freed when the class instance goes out of scope.
//    2. A resource shared among multiple subsystems; any subsystem may
//       outlive others, but the resource must persist until the last
//       subsystem is destroyed.
//    3. A cache storing recently used objects, where the cache itself
//       owns the objects (external code only observes them without
//       affecting their lifetime).
//    4. A factory function creates an object and returns it to the
//       caller, who takes over ownership.
//    5. A non-owning reference to an object whose lifetime is guaranteed
//       to exceed the reference.
//    6. A graph where parent nodes own child nodes, but child nodes need
//       back-pointers to parents without creating reference cycles.
//    7. A resource passed to a legacy C API expecting a raw pointer; the
//       API does not take ownership and will not free it.
//    8. A shared ownership scenario where one thread creates the
//       resource, and multiple threads read it; the resource is
//       destroyed only after all threads finish.
//    9. A polymorphic base class destructor that must be virtual; the
//       derived object is allocated with `new` and managed by a smart
//       pointer.
//   10. A non-owning pointer used in a performance-critical inner loop
//       where atomic reference-counting overhead is unacceptable.
//   11. A factory returning a pimpl handle; the handle is copyable and
//       shares ownership of the implementation.
//   12. A non-owning pointer stored in an object that is itself managed
//       by a std::shared_ptr, forming a potential cycle.
// ---------------------------------------------------------------------------
std::vector<std::string> smart_pointer_selection();
