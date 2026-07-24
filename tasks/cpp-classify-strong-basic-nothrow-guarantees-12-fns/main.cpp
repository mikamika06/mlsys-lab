// FIXED driver. Reproduces the 12 documented functions from task.md as REAL
// C++ (Widget's owned pointer is represented as an opaque "shadow" resource
// id rather than a raw `double*`, purely so a genuinely buggy candidate
// function can be exercised without corrupting real process memory -- the
// control flow / statement order of every function is otherwise exactly as
// documented).
//
// Ground truth is never hardcoded: for each function we (1) run it once
// with fault injection disabled to discover how many "risky" primitives
// (do_something / get_value / shadow_new, each modelling something that can
// throw) it invokes, then (2) run it once per risky call site with THAT
// call forced to throw, and OBSERVE the resulting object state -- unchanged
// at every site => strong, valid-but-possibly-changed at every site =>
// basic, invalid (dangling/double-freed) at any site => none, zero risky
// calls at all => nothrow.
#include <cstdio>
#include <functional>
#include <unordered_set>

#include "sol.hpp"

namespace {

struct Injected {};

long g_call_index = 0;
long g_throw_at = -1;

void fault_tick() {
    g_call_index++;
    if (g_call_index == g_throw_at) throw Injected{};
}

void do_something() { fault_tick(); }
int get_value() { fault_tick(); return 42; }

std::unordered_set<long> g_shadow_alive;
long g_shadow_next_id = 1;

long shadow_new(int n) {
    (void)n;
    fault_tick();  // models operator new[] possibly throwing std::bad_alloc
    long id = g_shadow_next_id++;
    g_shadow_alive.insert(id);
    return id;
}
void shadow_free(long id) {
    if (id == 0) return;
    g_shadow_alive.erase(id);  // erasing a not-present id is simply a no-op
                                // bookkeeping-wise (a REAL double free would
                                // corrupt memory; here it just fails to be
                                // "alive" afterward, which shadow_alive
                                // below turns into a detectable invariant
                                // violation instead of a crash).
}
bool shadow_alive(long id) { return id != 0 && g_shadow_alive.count(id) != 0; }
long shadow_clone(long src_id) { (void)src_id; return shadow_new(1); }

// Widget: `data` is the shadow-allocator id it owns (0 = owns nothing).
struct Widget {
    bool active = false;
    long data = 0;
    int count = 0;

    Widget() = default;
    ~Widget() { if (data != 0) shadow_free(data); }

    // Copy ctor/assign are GIVEN to provide only the basic guarantee.
    Widget(const Widget& o)
        : active(o.active), data(o.data != 0 ? shadow_clone(o.data) : 0), count(o.count) {}
    Widget& operator=(const Widget& o) {
        if (this == &o) return *this;
        if (data != 0) shadow_free(data);
        data = 0;
        data = (o.data != 0) ? shadow_clone(o.data) : 0;
        active = o.active;
        count = o.count;
        return *this;
    }

    void swap(Widget& o) noexcept {
        bool ta = active; long td = data; int tc = count;
        active = o.active; data = o.data; count = o.count;
        o.active = ta; o.data = td; o.count = tc;
    }
};

Widget make_test_widget(bool active_val, int count_val) {
    g_throw_at = -1;
    long id = shadow_new(64);
    Widget w;
    w.active = active_val;
    w.data = id;
    w.count = count_val;
    return w;
}

// ---- the 12 functions under test (faithful to task.md) ------------------
void f1(Widget& w) noexcept { w.active = false; }
void f2(Widget& w) { w.count++; do_something(); }
void f3(Widget& w) { Widget temp = w; temp.count++; w.swap(temp); }
void f4(Widget& w) { shadow_free(w.data); w.data = 0; do_something(); }
void f5(Widget& w) { shadow_free(w.data); do_something(); w.data = 0; }
void f6(Widget& w) { w.count = get_value(); w.active = true; }
void f7(Widget& w) { w.active = true; do_something(); w.count = 0; }
void f8(Widget& w) { long ptr = shadow_new(100); shadow_free(w.data); w.data = ptr; }
void f9(Widget& w) { do_something(); w.count++; }
void f10(Widget& w1, Widget& w2) { Widget temp = w1; w1 = w2; w2 = temp; }
void f11(Widget& w) { w.count = 0; }
void f12(Widget& w) { shadow_free(w.data); w.data = shadow_new(100); }

enum { NOTHROW = 0, STRONG = 1, BASIC = 2, NONE = 3 };

struct Snap { bool active; long data; int count; };
Snap snap(const Widget& w) { return {w.active, w.data, w.count}; }

// classify a single-Widget function by real fault injection. The seed
// widget starts with active=false so that functions which set active=true
// (f6, f7) produce an OBSERVABLE field change rather than accidentally
// reassigning the value it already had.
int classify1(const std::function<void(Widget&)>& fn) {
    Widget probe = make_test_widget(false, 5);
    g_throw_at = -1;
    g_call_index = 0;
    fn(probe);
    long N = g_call_index;
    if (N == 0) return NOTHROW;

    bool all_unchanged = true, all_valid = true;
    for (long t = 1; t <= N; t++) {
        Widget w = make_test_widget(false, 5);
        Snap before = snap(w);
        g_throw_at = t;
        g_call_index = 0;
        bool threw = false;
        try {
            fn(w);
        } catch (const Injected&) {
            threw = true;
        }
        if (!threw) continue;  // this call site wasn't actually reachable; ignore
        // A field can look numerically "unchanged" (e.g. a stale handle that
        // was never reassigned) while the resource it names was already
        // freed -- that is exactly the dangling-handle bug this classifier
        // must catch, so invalidity always overrides an apparent non-change.
        bool valid = (w.data == 0) || shadow_alive(w.data);
        bool unchanged = (w.active == before.active) && (w.data == before.data) && (w.count == before.count);
        if (!valid) all_valid = false;
        if (!unchanged) all_unchanged = false;
    }
    if (!all_valid) return NONE;
    if (all_unchanged) return STRONG;
    return BASIC;
}

// classify f10 (a two-Widget function) the same way.
int classify_f10() {
    Widget pa = make_test_widget(true, 5);
    Widget pb = make_test_widget(false, 9);
    g_throw_at = -1;
    g_call_index = 0;
    f10(pa, pb);
    long N = g_call_index;
    if (N == 0) return NOTHROW;

    bool all_unchanged = true, all_valid = true;
    for (long t = 1; t <= N; t++) {
        Widget a = make_test_widget(true, 5);
        Widget b = make_test_widget(false, 9);
        Snap ba = snap(a), bb = snap(b);
        g_throw_at = t;
        g_call_index = 0;
        bool threw = false;
        try {
            f10(a, b);
        } catch (const Injected&) {
            threw = true;
        }
        if (!threw) continue;
        bool valid = ((a.data == 0) || shadow_alive(a.data)) && ((b.data == 0) || shadow_alive(b.data));
        bool unchanged = (a.active == ba.active) && (a.data == ba.data) && (a.count == ba.count) &&
                          (b.active == bb.active) && (b.data == bb.data) && (b.count == bb.count);
        if (!valid) all_valid = false;
        if (!unchanged) all_unchanged = false;
    }
    if (!all_valid) return NONE;
    if (all_unchanged) return STRONG;
    return BASIC;
}

}  // namespace

int main() {
    int truth[12] = {
        classify1(f1),  classify1(f2), classify1(f3),  classify1(f4),
        classify1(f5),  classify1(f6), classify1(f7),  classify1(f8),
        classify1(f9),  classify_f10(), classify1(f11), classify1(f12),
    };

    int pred[12] = {};
    classify_guarantees(pred);

    int matches = 0;
    for (int i = 0; i < 12; i++) {
        int ok = (pred[i] == truth[i]) ? 1 : 0;
        matches += ok;
        printf("%d %d\n", i + 1, ok);
    }
    printf("matches %d\n", matches);
    return 0;
}
