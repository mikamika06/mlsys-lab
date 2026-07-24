#pragma once

// Instrumented probe, DEFINED in main.cpp: every special member logs one
// character to g_log --
//   'C' constructor  Probe(int)
//   'Y' copy constructor
//   'M' move constructor
//   'D' destructor
// -- via log_event(), so the exact sequence of events your code causes is
// directly observable.
extern char g_log[64];
extern int g_log_len;
void log_event(char c);

struct Probe {
    int id;
    explicit Probe(int id_);
    Probe(const Probe& other);
    Probe(Probe&& other) noexcept;
    ~Probe();
    // Copy/move ASSIGNMENT are deliberately unavailable -- only
    // construction and destruction are needed to reach the target
    // sequence, and disabling them rules out an accidental detour.
    Probe& operator=(const Probe&) = delete;
    Probe& operator=(Probe&&) = delete;
};

// Write the BODY of this function -- local Probe variables, copies, moves,
// nested scopes -- so that running it logs EXACTLY this event sequence, in
// order:
//
//     C Y M D D D
//
// i.e.: one Probe constructed (C); copy-constructed into a second Probe
// (Y); that second Probe move-constructed into a third (M); then all three
// destroyed (D D D), with the two innermost (the copy and its move target)
// destroyed before the first.
void reproduce_sequence();
