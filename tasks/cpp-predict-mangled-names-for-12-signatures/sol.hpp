#pragma once
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// Compute the Itanium C++ ABI mangled name for each signature string in
// `sigs`. Each signature has the exact format
//     "return_type function_name(param1_type, param2_type, ...)"
// with exactly one space between the return type and the function name,
// and ", " between parameter types (no return type is ever a pointer, no
// function name ever contains a space or parenthesis).
//
// Mangling rules (this restricted subset):
//   - A mangled name starts with "_Z", followed by the function name
//     encoded as its LENGTH (decimal, no leading zeros) followed by the
//     literal name, e.g. "foo" -> "3foo".
//   - The return type is NEVER part of the mangled name.
//   - Primitive type codes: void->v, bool->b, char->c, int->i, long->l,
//     float->f, double->d.
//   - A pointer type "T*" (optionally with whitespace before the `*`)
//     mangles as "P" followed by the mangled code of T (so "int*" ->
//     "Pi", "double*" -> "Pd").
//   - A function declared with an EMPTY parameter list, or with the
//     single parameter type "void", mangles as if it had exactly one
//     parameter of type void ("v") -- e.g. "void foo()" -> "_Z3foov".
//   - Otherwise, append the mangled code of every parameter type, in
//     order, with nothing between them.
//
// Example: "int bar(int, double)" -> "_Z3barid".
// ---------------------------------------------------------------------------
std::vector<std::string> mangle_signatures(const std::vector<std::string>& sigs);
