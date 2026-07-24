#pragma once

// ============================================================================
// Fixed instrumented result type (FIXED — do not modify). Every REAL copy
// or move construction bumps a global counter, so the counters are a direct
// trace of what the compiler actually generated for your `return`.
// ============================================================================
inline int g_copy_count = 0;
inline int g_move_count = 0;

struct Result {
    int a;
    double b;
    Result() = default;
    Result(int a_, double b_) : a(a_), b(b_) {}
    Result(const Result& o) : a(o.a), b(o.b) { ++g_copy_count; }
    Result(Result&& o) noexcept : a(o.a), b(o.b) { ++g_move_count; }
    Result& operator=(const Result&) = default;
    Result& operator=(Result&&) = default;
};

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// Return a Result holding (a1, b1) if cond is true, otherwise (a2, b2).
// Write it with a SINGLE named local `Result` declared once, assigned along
// whichever branch is taken, and exactly ONE `return` statement at the very
// end of the function — never two separate named locals each returned from
// their own `return` inside an if/else, and never a ternary combining two
// different objects in one return expression. That single-named-object,
// single-return shape is what named return value optimization (NRVO) needs
// to construct the result directly in the caller's storage, with zero
// copies and zero moves.
// ============================================================================
Result make_result(bool cond, int a1, double b1, int a2, double b2);
