#pragma once
#include <type_traits>
#include <utility>

// ============================================================================
// Fixed probe types (real C++). Each covers one overload-resolution edge
// case for a `serialize` member callable with exactly one int-convertible
// argument. Only data members affect layout; the member functions below are
// declared, never defined (never called) — only their SIGNATURES matter,
// for SFINAE to inspect.
// ============================================================================
struct DProbe1  { int x; void other(int); };                              // no serialize at all
struct DProbe2  { char x; void serialize(); };                            // 0-arg only, can't take 1
struct DProbe3  { int a; double b; void serialize(int); };                // exact 1-arg int
struct DProbe4  { char a, b; void serialize(int, int); };                 // requires 2, can't take 1
struct DProbe5  { void* p; void serialize(int, double = 0.0); };          // 1 required + 1 default
struct DProbe6  { short s; void serialize(void*); };                      // arg0 pointer: int can't convert
struct DProbe7  { float f; void serialize(double); };                     // int->double converts
struct DProbe8  { int x; void serialize(); void serialize(int); };        // overload set, 1-arg exists
struct DProbe9  { char c; void serialize(char*); };                       // arg0 pointer
struct DProbe10 { char c; void serialize(int = 0, int = 0, int = 0); };   // all defaulted, 1 is in range
struct DProbe11 { char c; void serialize(int = 0); };                     // 0 required, 1 accepted
struct DProbe12 { char c; void serialize(void*, int = 0); };              // arg0 pointer, 2nd defaulted

// ============================================================================
// LEARNER implements these 12 in solve.cpp.
//
// detect_DProbeN() must return whether DProbeN satisfies the "has_serialize"
// detection idiom: true iff
//     std::declval<DProbeN>().serialize(std::declval<int>())
// is a well-formed expression — i.e. some overload of `serialize` is
// callable with exactly one argument, and that argument's type accepts an
// `int` by implicit conversion (pointer parameters do NOT: an int does not
// implicitly convert to a pointer).
//
// Implement this with the classic detection idiom — a template
//     template <typename T, typename = void> struct has_serialize : std::false_type {};
//     template <typename T> struct has_serialize<T, std::void_t<decltype(...)>> : std::true_type {};
// defined once in solve.cpp — then instantiate has_serialize<DProbeN>::value
// once per wrapper below. (A template's specializations must live in the
// same translation unit as every place that instantiates it, so this trait
// belongs entirely in solve.cpp, not split into the shared header.)
// ============================================================================
bool detect_DProbe1();
bool detect_DProbe2();
bool detect_DProbe3();
bool detect_DProbe4();
bool detect_DProbe5();
bool detect_DProbe6();
bool detect_DProbe7();
bool detect_DProbe8();
bool detect_DProbe9();
bool detect_DProbe10();
bool detect_DProbe11();
bool detect_DProbe12();
