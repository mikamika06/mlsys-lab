#pragma once

// Appends one event string to the global lifecycle log (defined in main.cpp).
void log_event(const char* s);

// ============================================================================
// Fixed instrumented alternative types (FIXED — do not modify). Every
// construction/destruction logs an event, so the log is a direct trace of
// what your code actually did to their storage.
// ============================================================================
struct TypeA {
    int x;
    double y;
    TypeA() { log_event("ctor_TypeA"); }
    ~TypeA() { log_event("dtor_TypeA"); }
};

struct TypeB {
    char c;
    long l;
    TypeB() { log_event("ctor_TypeB"); }
    ~TypeB() { log_event("dtor_TypeB"); }
};

// ============================================================================
// A tagged union storage buffer for TypeA/TypeB (FIXED). The union itself
// does nothing on construction/destruction (`Storage`'s ctor/dtor are
// empty) — the ACTIVE member's lifetime is entirely YOUR responsibility,
// exactly like `std::variant`'s internals.
//
// active: 0 = nothing constructed yet, 1 = TypeA is the live member,
//         2 = TypeB is the live member.
// ============================================================================
struct MiniVariant {
    union Storage {
        Storage() {}
        ~Storage() {}
        TypeA a;
        TypeB b;
    };
    int active = 0;
    Storage storage;
};

// ============================================================================
// LEARNER implements these five in solve.cpp.
//
// variant_set_a / variant_set_b: if a member is currently active, destroy
// it FIRST (explicit destructor call — never skip this, even if the new
// type is the same as the old one), THEN placement-new the requested type
// into `v.storage`, then update `v.active`.
//
// variant_get_a / variant_get_b: log "access_TypeA"/"access_TypeB" if that
// type is the currently active member, otherwise log "invalid_access". Do
// NOT touch the storage's lifetime.
//
// variant_destroy: if a member is currently active, destroy it and set
// `v.active` back to 0. No-op if nothing is active.
// ============================================================================
void variant_set_a(MiniVariant& v);
void variant_set_b(MiniVariant& v);
void variant_get_a(MiniVariant& v);
void variant_get_b(MiniVariant& v);
void variant_destroy(MiniVariant& v);
