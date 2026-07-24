#pragma once

// Opaque wrapper around a float: its bits are PRIVATE. There is no
// operator<, operator>, or implicit conversion to float/bool -- so
// nothing you write in clamp_branchless can branch on a Guarded value's
// contents directly (the compiler rejects it, it isn't just a
// convention). The only ways to combine two Guarded values are the four
// functions below.
class Guarded {
public:
    static Guarded wrap(float v) { return Guarded(v); }

private:
    float v;
    friend float branchless_min(Guarded a, Guarded b);
    friend float branchless_max(Guarded a, Guarded b);
    friend float branchy_min(Guarded a, Guarded b);
    friend float branchy_max(Guarded a, Guarded b);
    explicit Guarded(float x) : v(x) {}
};

// BRANCHLESS primitives (DEFINED in main.cpp): each compiles to a
// single fmin/fmax-style select with no data-dependent control flow.
// Free to call as many times as you like -- they never touch the
// harness's branch counter.
float branchless_min(Guarded a, Guarded b);
float branchless_max(Guarded a, Guarded b);

// ESCAPE HATCH (DEFINED in main.cpp): computes the exact same result as
// branchless_min / branchless_max, but internally with a real `if`, and
// every call increments a counter the harness prints at the end. It
// exists only so code that insists on branching still compiles --
// calling it means the implementation is not branchless.
float branchy_min(Guarded a, Guarded b);
float branchy_max(Guarded a, Guarded b);

// LEARNER IMPLEMENTS.
//
// Return x clamped into [lo, hi] -- i.e. max(lo, min(x, hi)) -- using
// ONLY branchless_min / branchless_max (never branchy_min / branchy_max,
// which the harness is counting).
float clamp_branchless(Guarded x, Guarded lo, Guarded hi);
