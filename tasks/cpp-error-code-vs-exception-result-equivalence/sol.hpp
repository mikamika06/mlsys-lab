#pragma once
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// PROVIDED (do not change): a naive std::expected<double,int> stand-in
// (no union, so its real compiled size shows exactly what tag+payload
// storage costs), and a real exception type carrying the same error code.
// ---------------------------------------------------------------------------
struct NaiveExpected {
    bool   has_value;
    double val;
    int    err;
};

// Thrown by compute_exceptions() on failure: a genuine C++ exception type
// carrying the same error code compute_expected() would return in `err`.
struct OpFailure {
    int err;
    explicit OpFailure(int e) : err(e) {}
};

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS all three.
//
// `ops` is a sequence of operation strings, each either "add X", "sub X"
// (X a floating-point literal), "div0", or "overflow". Both functions run
// the SAME state machine starting from state = 0.0:
//   "add X"    -> state += X
//   "sub X"    -> state -= X
//   "div0"     -> fails with error code 1
//   "overflow" -> fails with error code 2
//
// compute_expected: the error-code / std::expected model.
//   - On success (every op consumed without failing): return
//     {has_value: true, val: final state, err: <don't care>}.
//   - On failure: return {has_value: false, val: <don't care>, err: code},
//     stopping immediately -- ops after the failing one are never applied.
//
// compute_exceptions: the exception model, using REAL C++ throw/catch
// (stack unwinding), not a disguised error return.
//   - On success: return the final state as a double.
//   - On failure: `throw OpFailure(code)` immediately -- ops after the
//     failing one are never applied.
//
// naive_expected_size: return sizeof(NaiveExpected) as compiled by the
// real compiler (there is no separate spec to hand-compute against).
// ---------------------------------------------------------------------------
NaiveExpected compute_expected(const std::vector<std::string>& ops);
double        compute_exceptions(const std::vector<std::string>& ops);
long          naive_expected_size();
