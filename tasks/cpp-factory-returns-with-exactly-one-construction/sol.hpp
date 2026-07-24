#pragma once

// ---------------------------------------------------------------------------
// PROVIDED (do not change): an instrumented Probe type. Every construction
// path bumps its own global counter, so the driver can observe exactly how
// many direct constructions, copies, and moves actually happened.
// ---------------------------------------------------------------------------
struct Probe {
    int tag;
    double payload;

    explicit Probe(int tag_, double payload_);  // "direct" construction
    Probe(const Probe& other);                  // counted copy
    Probe(Probe&& other) noexcept;              // counted move
    Probe& operator=(const Probe&) = delete;
    Probe& operator=(Probe&&) = delete;
    ~Probe();
};

// PROVIDED (defined in main.cpp); reset to 0 before every call.
extern long g_direct_count;
extern long g_copy_count;
extern long g_move_count;

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// A factory that hands back a freshly built Probe(tag, payload) BY VALUE,
// as a prvalue, so that C++17's GUARANTEED copy elision constructs it
// directly in the caller's storage. Done correctly, exactly ONE constructor
// call happens in total across the whole call -- the direct
// Probe(tag, payload) constructor (g_direct_count == 1) -- with
// g_copy_count == 0 and g_move_count == 0.
//
// The only way to break this is to introduce an intermediate named Probe
// object and hand back a copy (or move) of it through some other
// expression -- that is a real, non-elidable extra construction.
// ---------------------------------------------------------------------------
Probe make_probe(int tag, double payload);
