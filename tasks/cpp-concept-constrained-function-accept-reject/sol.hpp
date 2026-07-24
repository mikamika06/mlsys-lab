#pragma once
// Six custom probe types, each shaped around a different operator+ story.
// The other six probes are built-in types: int, double, float, bool, long,
// char.

struct SelfAdd {
    int v;
    SelfAdd operator+(const SelfAdd& o) const { return SelfAdd{v + o.v}; }
};

struct DifferentReturn {
    int v;
    // Returns long, not DifferentReturn.
    long operator+(const DifferentReturn& o) const { return v + o.v; }
};

struct MissingAdd {
    int v;
    // No operator+ at all.
};

struct AmbiguousAdd {
    int v;
    AmbiguousAdd operator+(const AmbiguousAdd&) const { return AmbiguousAdd{v}; }
};
// A free operator+ with an equally-good conversion sequence for `x + x`
// makes overload resolution ambiguous between it and the member above.
AmbiguousAdd operator+(AmbiguousAdd a, AmbiguousAdd b);

struct DeletedAdd {
    int v;
    DeletedAdd operator+(const DeletedAdd&) const = delete;
};

struct MixedOnly {
    int v;
    // operator+ exists only for (MixedOnly, int), not (MixedOnly, MixedOnly).
    MixedOnly operator+(int rhs) const { return MixedOnly{v + rhs}; }
};

// Fill out[0..11] with 1 if the corresponding probe type T satisfies the
// C++20 concept
//   template<class T> concept Acceptable =
//       requires(T x) { { x + x } -> std::same_as<T>; };
// and 0 otherwise, testing these 12 types in exactly this order:
//   0 int, 1 double, 2 float, 3 bool, 4 long, 5 char,
//   6 SelfAdd, 7 DifferentReturn, 8 MissingAdd, 9 AmbiguousAdd,
//   10 DeletedAdd, 11 MixedOnly
void classify_accepts(int out[12]);
