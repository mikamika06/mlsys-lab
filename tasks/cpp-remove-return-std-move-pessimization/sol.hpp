#pragma once

// ---------------------------------------------------------------------------
// PROVIDED (do not change): an instrumented type that counts its own real
// copies and moves through actual constructor calls.
// ---------------------------------------------------------------------------
struct Widget {
    int value;
    Widget();
    explicit Widget(int v);
    Widget(const Widget&);       // bumps g_copy_count
    Widget(Widget&&) noexcept;   // bumps g_move_count
    Widget& operator=(const Widget&) = delete;
    Widget& operator=(Widget&&)      = delete;
};

extern long g_copy_count;
extern long g_move_count;

// ---------------------------------------------------------------------------
// LEARNER FIXES a bug.
//
// Build a local Widget holding `v` and return it BY VALUE so that Named
// Return Value Optimization (NRVO) can construct it directly in the
// caller's return slot, with zero copies and zero moves. The shipped
// implementation writes `return std::move(w);`, which casts the local to
// an rvalue reference: the return expression is no longer a plain local
// variable identifier, so NRVO is inhibited and the compiler is forced to
// move-construct the result instead (g_move_count would be bumped by 1).
//
// Fix it: return the plain local variable, not std::move(it).
// ---------------------------------------------------------------------------
Widget make_widget(int v);
