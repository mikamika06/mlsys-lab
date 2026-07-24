#pragma once

// Predict, for each of the 15 documented C++ snippets in task.md, whether
// clang's UndefinedBehaviorSanitizer (-fsanitize=undefined) reports it (1)
// or not (0). out[i] holds your prediction for snippet i+1.
void predict_ubsan_flags(int out[15]);
