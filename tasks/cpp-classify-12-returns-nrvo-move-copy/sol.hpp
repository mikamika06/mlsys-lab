#pragma once
#include <string>

// Classify how each of the 12 return statements f1..f12 (see task.md)
// actually returns its `T` by value. out[i] (for snippet i+1, i.e. out[0]
// is f1 ... out[11] is f12) must be exactly one of:
//   "rvo"  - the returned expression is a prvalue; C++17 guarantees the
//            construction happens directly in the caller's storage (no
//            copy or move constructor call at all).
//   "nrvo" - the returned expression names a local automatic-storage-
//            duration object on a single return path; elision into the
//            caller's storage is permitted (and, on this compiler,
//            observed) with no copy or move constructor call.
//   "move" - a move constructor call is required (e.g. std::move, or an
//            implicitly-movable entity such as a by-value parameter that
//            isn't itself NRVO-eligible).
//   "copy" - a copy constructor call is required (the source is not an
//            automatic-storage-duration object owned by this function:
//            a reference, a global, a static, a dereferenced pointer, a
//            ternary between two distinct named locals, or a subobject).
//
// Also return your prediction for sizeof(T) under LP64, where
//   struct T { char c; double d; int i; };
void predict_return_kinds(std::string out[12]);
int predict_struct_size();
