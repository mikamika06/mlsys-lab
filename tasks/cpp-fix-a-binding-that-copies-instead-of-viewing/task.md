## Context

When interfacing C++ data structures with Python via libraries like
pybind11 or the buffer protocol, creating zero-copy views around existing
C++ buffers is essential for performance. A common but subtle bug: a
binding allocates a fresh buffer and copies the payload into it "to be
safe". Reads through the result look correct, but the binding has silently
become an *owning copy* instead of a *view* — writes through it never reach
the original buffer, and `np.shares_memory(original, result)` is `False`
when it should be `True`.

Given a single contiguous allocation laid out as a header of `header_size`
bytes followed immediately by a payload of `n` doubles, a correct zero-copy
binding returns a pointer that is *literally* `buf + header_size`
reinterpreted as `double*` — the same memory, not a duplicate.

## Task

Fix `view_payload` in `solve.cpp`:

```cpp
double* view_payload(unsigned char* buf, int header_size, int n);
```

It must return `reinterpret_cast<double*>(buf + header_size)` — a pointer
into `buf`'s own storage. Do not allocate a new buffer or copy any bytes.

The fixed driver in `main.cpp`:
1. builds a `buf` with a filler header followed by `n` deterministic
   payload doubles,
2. calls `view_payload` and prints what it reads back,
3. **writes** through the returned pointer, then reads directly out of
   `buf` at the same offset and prints that,
4. prints whether the returned pointer is pointer-equal to
   `buf + header_size` (the zero-copy check, standing in for
   `np.shares_memory`).

## Example

```cpp
double* view = view_payload(buf, header_size, n);
view[0] = 99.0;
// A correct view: *(double*)(buf + header_size) is now 99.0 too.
// A buggy copy:    *(double*)(buf + header_size) is unchanged.
```

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with real
`clang++ -O2 -std=c++20`, runs it, and compares stdout byte-for-byte against
the reference build (`exact_match == 1.0`). The starter allocates a fresh
`double[n]` and `memcpy`s the payload into it: the first printed line
(plain reads) happens to still match, but writing through the returned
pointer never reaches `buf`, so the "direct-after-write" line and the
pointer-equality line both fail.
