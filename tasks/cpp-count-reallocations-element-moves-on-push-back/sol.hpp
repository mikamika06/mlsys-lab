#pragma once
#include <cstddef>

// ---------------------------------------------------------------------------
// PROVIDED (do not change): the real element type under this compiler's
// LP64 ABI, and a counted move constructor so the driver can observe
// exactly how many element moves your growable array performs. Element is
// NOT copyable, so the only way to relocate one is to move-construct it --
// exactly what a real std::vector<Element> does on reallocation for a
// noexcept-movable type.
// ---------------------------------------------------------------------------
struct Element {
    short   type;
    double* data;
    long    sizes[3];

    Element();                            // zero-initializes a fresh Element
    Element(const Element&)            = delete;
    Element& operator=(const Element&) = delete;
    Element(Element&& other) noexcept;    // counted move: bumps g_move_count
    Element& operator=(Element&&)      = delete;
};

// PROVIDED (defined in main.cpp): bumped once per Element move-construction.
extern long g_move_count;

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// Simulate a std::vector<Element>-style growable array performing N
// push_back operations of freshly constructed temporary Elements, using
// the exact growth policy real std::vector implementations follow:
//
//   - Start with size = 0, capacity = initial_capacity, no allocation if
//     initial_capacity == 0.
//   - Before each push, if size == capacity, reallocate: allocate a new
//     raw buffer (via ::operator new, NOT `new Element[...]`) of
//     new_capacity = (capacity == 0 ? 1 : capacity * growth_factor)
//     elements, then move-construct (placement-new with Element&&) every
//     existing element from the old buffer into the new one in order,
//     destroy the old elements, and release the old raw buffer
//     (::operator delete).
//   - After ensuring capacity, construct a temporary Element and
//     move-construct it into slot `size` of the buffer, then size += 1.
//   - Before returning, destroy every live element and release the raw
//     buffer -- no leaks.
//
// Every move-construction, whether a reallocation move or the final
// insertion move, MUST go through Element's real move constructor (so it
// increments g_move_count itself) -- never hand-count moves yourself.
//
// Return the number of reallocations performed across all N pushes.
// (g_move_count, read by the driver right after this call returns,
// reports the total move count.)
// ---------------------------------------------------------------------------
long simulate_vector_pushes(int N, int initial_capacity, int growth_factor);
