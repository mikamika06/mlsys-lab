// FIXED driver. Runs 24 real C++ functions, each reproducing one of the
// lifetime snippets documented in task.md, using an instrumented `Gadget`
// whose ctors/dtor record its `this` address in a global "currently alive"
// set. At each snippet's MARK point we ask the set "is this address alive
// right now?" -- pure pointer-identity comparison, never a read through a
// possibly-dangling pointer, so there is no UB even for the dangling cases.
// That gives a REAL, runtime-computed ground truth (never hardcoded) to
// compare the candidate's predictions against.
#include <cstdio>
#include <memory>
#include <new>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sol.hpp"

static std::unordered_set<const void*> g_alive;
static const void* g_last_default_ctor_addr = nullptr;

struct Gadget {
    int id;
    void* buffer;
    Gadget() : id(0), buffer(nullptr) { g_alive.insert(this); g_last_default_ctor_addr = this; }
    Gadget(const Gadget& o) : id(o.id), buffer(o.buffer) { g_alive.insert(this); }
    Gadget(Gadget&& o) noexcept : id(o.id), buffer(o.buffer) { g_alive.insert(this); }
    Gadget& operator=(const Gadget&) = default;
    Gadget& operator=(Gadget&&) = default;
    ~Gadget() { g_alive.erase(this); }
};

static bool alive(const void* p) { return g_alive.count(p) != 0; }

// 1. void f() { Gadget g; /* MARK */ }
static bool truth_1() {
    Gadget g;
    return alive(&g);
}

// 2. void f() { { Gadget g; } /* MARK */ }
static bool truth_2() {
    const void* addr;
    { Gadget g; addr = &g; }
    return alive(addr);
}

// 3. Gadget* p = new Gadget; delete p; /* MARK */
static bool truth_3() {
    Gadget* p = new Gadget;
    delete p;
    return alive(p);
}

// 4. Gadget* p = new Gadget; /* MARK */ delete p;
static bool truth_4() {
    Gadget* p = new Gadget;
    bool result = alive(p);
    delete p;
    return result;
}

// 5. const Gadget& ref = Gadget(); /* MARK */  (const& extends temporary)
static bool truth_5() {
    const Gadget& ref = Gadget();
    return alive(&ref);
}

// 6. Gadget&& ref = Gadget(); /* MARK */  (&& also extends temporary)
static bool truth_6() {
    Gadget&& ref = Gadget();
    return alive(&ref);
}

// 7. Gadget* f() { Gadget g; return &g; } void g() { Gadget* p = f(); /* MARK */ }
static Gadget* helper7() {
    Gadget g;
    return &g;
}
static bool truth_7() {
    Gadget* p = helper7();
    return alive(p);
}

// 8. const Gadget& f() { return Gadget(); } void g() { const Gadget& p = f(); /* MARK */ }
static const Gadget& helper8() {
    return Gadget();
}
static bool truth_8() {
    const Gadget& p = helper8();
    return alive(&p);
}

// 9. void f() { static Gadget g; /* MARK */ }
static bool truth_9() {
    static Gadget g;
    return alive(&g);
}

// 10. void f() { Gadget g; auto l = [&g]() { /* MARK */ }; l(); }
static bool truth_10() {
    Gadget g;
    bool result = false;
    auto l = [&]() { result = alive(&g); };
    l();
    return result;
}

// 11. auto f() { Gadget g; return [&g]() { /* MARK */ }; } void h() { f()(); }
static auto helper11() {
    Gadget g;
    return [&g]() -> bool { return alive(&g); };
}
static bool truth_11() {
    return helper11()();
}

// 12. void f() { auto p = std::make_unique<Gadget>(); /* MARK */ }
static bool truth_12() {
    auto p = std::make_unique<Gadget>();
    return alive(p.get());
}

// 13. void f() { auto p = std::make_unique<Gadget>(); p = nullptr; /* MARK */ }
static bool truth_13() {
    auto p = std::make_unique<Gadget>();
    const void* addr = p.get();
    p = nullptr;
    return alive(addr);
}

// 14. void f() { auto p1 = std::make_shared<Gadget>(); { auto p2 = p1; } /* MARK */ }
static bool truth_14() {
    auto p1 = std::make_shared<Gadget>();
    { auto p2 = p1; (void)p2; }
    return alive(p1.get());
}

// 15. auto p1 = make_shared<Gadget>(); weak_ptr<Gadget> w = p1; p1.reset(); /* MARK (target = object w points to) */
static bool truth_15() {
    auto p1 = std::make_shared<Gadget>();
    std::weak_ptr<Gadget> w = p1;
    const void* addr = p1.get();
    p1.reset();
    (void)w;
    return alive(addr);
}

// 16. Gadget* g; void init(){g=new Gadget;} void destroy(){delete g;} int main(){init();destroy();/* MARK */}
static Gadget* g_global16 = nullptr;
static void init16() { g_global16 = new Gadget; }
static void destroy16() { delete g_global16; }
static bool truth_16() {
    init16();
    destroy16();
    return alive(g_global16);
}

// 17. void f(const Gadget& g) { /* MARK */ }; void h() { f(Gadget()); }
static bool helper17(const Gadget& g) { return alive(&g); }
static bool truth_17() {
    return helper17(Gadget());
}

// 18. void f() { Gadget g; std::move(g); /* MARK */ }
static bool truth_18() {
    Gadget g;
    (void)std::move(g);
    return alive(&g);
}

// 19. placement-new + explicit dtor call, then MARK
static bool truth_19() {
    alignas(Gadget) char buf[sizeof(Gadget)];
    Gadget* p = new (buf) Gadget;
    p->~Gadget();
    return alive(p);
}

// 20. Gadget g; void f() { /* MARK */ }  (namespace-scope global)
static Gadget g_global20;
static bool truth_20() {
    return alive(&g_global20);
}

// 21. void f() { std::vector<Gadget> v; v.push_back(Gadget()); /* MARK (target = the temporary) */ }
static bool truth_21() {
    std::vector<Gadget> v;
    v.push_back(Gadget());
    return alive(g_last_default_ctor_addr);
}

// 22. void f() { std::vector<Gadget> v(1); Gadget* p = &v[0]; v.clear(); /* MARK */ }
static bool truth_22() {
    std::vector<Gadget> v(1);
    Gadget* p = &v[0];
    const void* addr = p;
    v.clear();
    return alive(addr);
}

// 23. void f() { thread_local Gadget g; /* MARK */ }
static bool truth_23() {
    thread_local Gadget g;
    return alive(&g);
}

// 24. void f() { for(int i=0;i<1;++i) { Gadget g; /* MARK */ } }
static bool truth_24() {
    bool result = false;
    for (int i = 0; i < 1; ++i) {
        Gadget g;
        result = alive(&g);
    }
    return result;
}

int main() {
    bool truth[24] = {
        truth_1(),  truth_2(),  truth_3(),  truth_4(),  truth_5(),  truth_6(),
        truth_7(),  truth_8(),  truth_9(),  truth_10(), truth_11(), truth_12(),
        truth_13(), truth_14(), truth_15(), truth_16(), truth_17(), truth_18(),
        truth_19(), truth_20(), truth_21(), truth_22(), truth_23(), truth_24(),
    };
    int true_size = (int)sizeof(Gadget);

    bool pred[24] = {};
    int pred_size = predict_lifetimes(pred);

    int matches = 0;
    for (int i = 0; i < 24; i++) {
        int ok = (pred[i] == truth[i]) ? 1 : 0;
        matches += ok;
        printf("%d %d\n", i + 1, ok);
    }
    printf("matches %d\n", matches);
    printf("size_pred %d\n", pred_size);
    printf("size_true %d\n", true_size);
    printf("size_ok %d\n", pred_size == true_size ? 1 : 0);
    return 0;
}
