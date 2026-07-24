#pragma once
// Predict, for each of the 24 documented C++ lifetime snippets (see
// task.md), whether the marked target object is within its lifetime
// (true) or already outside it (false) at the point marked /* MARK */ in
// that snippet — out[i] holds the prediction for snippet i+1 (i.e. out[0]
// is snippet 1, out[23] is snippet 24).
//
// Also return your prediction for sizeof(Gadget) under the LP64 ABI, where
// Gadget is `struct Gadget { int id; void* buffer; };` (the exact struct
// main.cpp uses, instrumented to track object lifetimes for real).
int predict_lifetimes(bool out[24]);
