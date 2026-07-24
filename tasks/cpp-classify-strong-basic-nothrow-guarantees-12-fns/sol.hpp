#pragma once

// Predict the STRONGEST exception-safety guarantee that each of the 12
// documented functions (see task.md) provides for its Widget parameter(s).
//
// Guarantee codes:
//   0 = nothrow  (never calls anything that can fail)
//   1 = strong   (on throw, every touched object is left EXACTLY as it was)
//   2 = basic    (on throw, every touched object stays in a valid state,
//                 but possibly different from before the call)
//   3 = none     (on throw, some object can end up in an invalid/dangling
//                 state)
//
// out[i] holds your prediction for function i+1 (out[0] = f1, ..., out[11] = f12).
void classify_guarantees(int out[12]);
