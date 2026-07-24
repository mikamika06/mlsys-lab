#pragma once
#include <string>

// ---------------------------------------------------------------------------
// PROVIDED infrastructure (do not change): a global construction/destruction
// event log, and a Probe type that records its own birth and death in it.
//
//   * When a Probe with id N is constructed  -> "C<N> " is appended.
//   * When a Probe with id N is destroyed    -> "D<N> " is appended.
//
// Probe is deliberately NON-copyable and NON-movable, so the only legal way to
// place many of them in a raw arena buffer is placement-new, and the only way
// to end their lifetimes is an explicit destructor call.
// ---------------------------------------------------------------------------
inline std::string g_events;

struct Probe {
    char   tag;
    int    id;
    double weight;

    explicit Probe(int i) : tag('P'), id(i), weight(0.0) {
        g_events += "C" + std::to_string(i) + " ";
    }
    ~Probe() {
        g_events += "D" + std::to_string(id) + " ";
    }

    Probe(const Probe&)            = delete;
    Probe(Probe&&)                 = delete;
    Probe& operator=(const Probe&) = delete;
    Probe& operator=(Probe&&)      = delete;
};

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// Build a scoped arena that holds one Probe per id in ids[0..n):
//   1. reserve a single raw byte buffer big enough for all n Probes,
//   2. construct a Probe for each id in FORWARD order (placement-new),
//   3. tear the arena down, destroying every Probe in strict LIFO order
//      (reverse of construction),
//   4. return the arena footprint in bytes: n * sizeof(Probe).
//
// All construction and destruction must happen before returning, so the global
// event log holds the full "C.../D..." sequence once the call completes.
// ---------------------------------------------------------------------------
long run_scoped_arena(const int* ids, int n);
