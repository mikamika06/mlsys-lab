#pragma once
#include <cstddef>
#include <new>

// ---------------------------------------------------------------------------
// FIXED, do not modify: an element type that tracks its own lifetime.
// Every construction bumps `alive` and `ctor_calls`; every destruction bumps
// `dtor_calls` and drops `alive`. A container that manages lifetimes correctly
// must leave `alive == 0` after it is destroyed and keep ctor_calls == dtor_calls.
// ---------------------------------------------------------------------------
struct Tracked {
    long value;
    static inline long alive = 0;       // currently live Tracked objects
    static inline long ctor_calls = 0;  // total constructions (any ctor)
    static inline long dtor_calls = 0;  // total destructions

    Tracked() : value(0) { ++alive; ++ctor_calls; }
    explicit Tracked(long v) : value(v) { ++alive; ++ctor_calls; }
    Tracked(const Tracked& o) : value(o.value) { ++alive; ++ctor_calls; }
    ~Tracked() { --alive; ++dtor_calls; }
};

// ---------------------------------------------------------------------------
// SmallVector: a growable buffer of Tracked with INLINE storage for up to CAP
// elements. While size() <= CAP the elements live inside `inbuf` (no heap
// allocation). The first push past CAP must "spill" onto the heap.
//
// The storage layout below is FIXED. You implement every member function in
// solve.cpp using placement new to construct elements and manual destructor
// calls to destroy them. Each element must be constructed exactly once and
// destroyed exactly once (no leaks, no double-free, no double-construct).
// ---------------------------------------------------------------------------
struct SmallVector {
    static constexpr int CAP = 4;

    alignas(Tracked) unsigned char inbuf[sizeof(Tracked) * CAP];
    Tracked* data;   // points at inbuf while inline, at heap memory after a spill
    int sz;          // number of live elements
    int cap;         // current capacity (CAP while inline)

    SmallVector();
    ~SmallVector();

    // Append a Tracked{v}. If sz == cap, first grow capacity (doubling) by
    // moving/copy-constructing the existing elements into the new storage,
    // destroying the old ones, and freeing old heap memory (if any).
    void push_back(long v);

    long sum() const;       // sum of every element's `value`
    int size() const;       // number of live elements
    bool spilled() const;   // true iff storage is currently on the heap
};
