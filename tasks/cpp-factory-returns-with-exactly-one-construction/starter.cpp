#include "sol.hpp"

// TODO: return the Probe directly as a prvalue (`return Probe(tag, payload);`)
// so guaranteed copy elision builds it in place with a single construction.
//
// This version instead builds a named local first and hands back an
// explicit copy of it -- a real, non-elidable extra construction (copying
// from an already-existing named object is never something the "as-if"
// elision rules are allowed to skip).
Probe make_probe(int tag, double payload) {
    Probe local(tag, payload);
    return Probe(local);
}
