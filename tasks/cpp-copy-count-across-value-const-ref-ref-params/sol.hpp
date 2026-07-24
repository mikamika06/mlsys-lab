#pragma once

// ============================================================================
// Fixed instrumented probe (FIXED — do not modify these definitions).
// g_copy_count is bumped by the copy constructor every time a Probe is
// copy-constructed. `inline` gives one shared instance across every
// translation unit.
// ============================================================================
inline int g_copy_count = 0;

struct Probe {
    char label;
    int count;
    double data[3];

    Probe() : label('x'), count(0), data{1.0, 2.0, 3.0} {}
    Probe(const Probe& other)
        : label(other.label), count(other.count),
          data{other.data[0], other.data[1], other.data[2]} {
        ++g_copy_count;
    }
};

// Fixed overloads (defined in main.cpp) — the number of copies that happen
// is decided entirely by HOW you call these, not by anything inside them.
void process_value(Probe p);
void process_const_ref(const Probe& p);
void process_ref(Probe& p);

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// Run, IN ORDER, the four scenarios below against a single `Probe obj` you
// construct yourself. Before EACH call, reset g_copy_count to 0; right
// after it, write the count observed into out[i]:
//
//   out[0] = copies from process_value(obj)        obj: lvalue, by value
//   out[1] = copies from process_const_ref(obj)     obj: lvalue, by const&
//   out[2] = copies from process_ref(obj)           obj: lvalue, by &
//   out[3] = copies from process_value(Probe{})     a fresh PRVALUE, by value
//
// Scenario 4 must construct the temporary directly in the call expression
// (`process_value(Probe{})`) — NOT bind it to a named variable first. Under
// C++17 guaranteed copy elision, a prvalue passed by value materializes
// straight into the parameter with no copy or move; binding it to a name
// first turns it into an lvalue, and passing THAT along forces a real copy.
// ============================================================================
void run_scenarios(int out[4]);
