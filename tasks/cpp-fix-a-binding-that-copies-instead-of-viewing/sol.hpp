#pragma once

// `buf` points at a single contiguous allocation: a header of exactly
// `header_size` bytes, followed immediately by a payload of `n` doubles.
//
// Return a ZERO-COPY VIEW into that same memory — i.e. exactly
// reinterpret_cast<double*>(buf + header_size) — so that:
//   - reading through the returned pointer sees the current payload bytes
//   - WRITING through the returned pointer is visible through `buf` too,
//     because it is the same underlying storage, not a copy
//
// Do NOT allocate a new `double[n]` and memcpy the payload into it — that
// is exactly the bug this task asks you to fix, and it breaks the
// zero-copy contract this function exists to provide (pybind11 / NumPy
// `frombuffer`-style zero-copy interop).
double* view_payload(unsigned char* buf, int header_size, int n);
