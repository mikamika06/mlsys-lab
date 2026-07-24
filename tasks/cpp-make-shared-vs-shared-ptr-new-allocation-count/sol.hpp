#pragma once
#include <memory>

// A plain payload type, big enough to make the extra control-block
// allocation of shared_ptr<T>(new T) visible against sizeof(T) alone.
struct Payload {
    int a;
    double b;
    char c;
};

// Construct a std::shared_ptr<Payload> the way `use_make_shared` selects:
//   true  -> std::make_shared<Payload>(...)          ONE heap allocation
//            (control block and Payload share a single allocation)
//   false -> std::shared_ptr<Payload>(new Payload(...))  TWO allocations
//            (one for Payload, one for the separate control block)
// Fill the Payload's fields with a, b, c either way. The driver measures
// the REAL number and total size of heap allocations this call makes
// through an instrumented global operator new -- you don't count or
// predict anything, you just have to call the RIGHT construction for the
// flag.
std::shared_ptr<Payload> make_payload(bool use_make_shared, int a, double b, char c);
