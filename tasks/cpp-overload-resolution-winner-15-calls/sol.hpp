#pragma once

// Predict, for each of the 15 documented overload-resolution scenarios in
// task.md, which of that scenario's two candidate overloads (tag 0 or
// tag 1) real C++ overload resolution actually selects.
// out[i] holds your prediction for scenario i+1 (out[0] = scenario 1, ...,
// out[14] = scenario 15).
void predict_overload_winners(int out[15]);
