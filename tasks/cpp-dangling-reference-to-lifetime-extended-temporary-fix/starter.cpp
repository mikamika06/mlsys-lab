#include "sol.hpp"

// BROKEN: a helper returns a const& bound directly to a temporary Result.
// The temporary's (extended) lifetime ends when the helper itself returns,
// so the reference get_result() receives is already dangling — reading
// through it afterward is undefined behaviour, and the stack scratch below
// is enough to make that observable.
static const Result& make_dangling(int id, float val) {
    return Result{id, val};
}

Result get_result(int id, float val) {
    const Result& danger = make_dangling(id, val);
    volatile int junk[64];
    for (int i = 0; i < 64; ++i) junk[i] = 0xDEADBEEF ^ i;
    return danger;
}
