#include <new>
#include "sol.hpp"

static void destroy_active(MiniVariant& v) {
    if (v.active == 1) {
        v.storage.a.~TypeA();
    } else if (v.active == 2) {
        v.storage.b.~TypeB();
    }
    v.active = 0;
}

void variant_set_a(MiniVariant& v) {
    destroy_active(v);
    new (&v.storage.a) TypeA();
    v.active = 1;
}

void variant_set_b(MiniVariant& v) {
    destroy_active(v);
    new (&v.storage.b) TypeB();
    v.active = 2;
}

void variant_get_a(MiniVariant& v) {
    if (v.active == 1) {
        log_event("access_TypeA");
    } else {
        log_event("invalid_access");
    }
}

void variant_get_b(MiniVariant& v) {
    if (v.active == 2) {
        log_event("access_TypeB");
    } else {
        log_event("invalid_access");
    }
}

void variant_destroy(MiniVariant& v) {
    destroy_active(v);
}
