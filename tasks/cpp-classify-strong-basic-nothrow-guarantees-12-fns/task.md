## Context

Exception safety is classified into four levels of guarantee:

1. **Nothrow**: the function never calls anything that can fail — it is
   guaranteed to complete.
2. **Strong**: if an exception escapes, every object the function touched is
   left in **exactly** the state it was in before the call (commit-or-rollback).
3. **Basic**: if an exception escapes, every touched object is still in a
   **valid** state (no dangling handles, no double frees), but its value may
   have changed from before the call.
4. **None**: if an exception escapes, some object can be left **invalid**
   (e.g. holding a handle to memory that was already freed).

All 12 functions below take a `Widget`:

```cpp
struct Widget {
    bool active;
    long data;   // an owned resource handle; 0 means "owns nothing"
    int  count;
    // copy ctor / operator= are given and provide only the BASIC guarantee
    // swap(Widget&) is given and is noexcept
};
```

`do_something()` and `get_value()` are external calls that may throw.
`shadow_new(n)` models `new[]` (may throw `std::bad_alloc`) and returns a
fresh handle; `shadow_free(id)` models `delete[]` (can never throw).

## Task

Determine the **strongest** guarantee each function provides for its
`Widget` argument(s):

```cpp
void f1(Widget& w) noexcept { w.active = false; }
void f2(Widget& w) { w.count++; do_something(); }
void f3(Widget& w) { Widget temp = w; temp.count++; w.swap(temp); }
void f4(Widget& w) { shadow_free(w.data); w.data = 0; do_something(); }
void f5(Widget& w) { shadow_free(w.data); do_something(); w.data = 0; }
void f6(Widget& w) { w.count = get_value(); w.active = true; }
void f7(Widget& w) { w.active = true; do_something(); w.count = 0; }
void f8(Widget& w) { long ptr = shadow_new(100); shadow_free(w.data); w.data = ptr; }
void f9(Widget& w) { do_something(); w.count++; }
void f10(Widget& w1, Widget& w2) { Widget temp = w1; w1 = w2; w2 = temp; }
void f11(Widget& w) { w.count = 0; }
void f12(Widget& w) { shadow_free(w.data); w.data = shadow_new(100); }
```

Implement `classify_guarantees(int out[12])`, filling `out[i]` with your
prediction for function `i+1` using the codes `0 = nothrow`, `1 = strong`,
`2 = basic`, `3 = none`.

Reasoning hints, by pattern:
- If nothing risky is ever called, it's `nothrow` (e.g. a plain field write).
- "Do the risky thing first, mutate only after it succeeds" (`f6`, `f8`, `f9`)
  gives `strong` — if the risky call throws, nothing has been touched yet.
- Copy-and-swap (`f3`) is `strong` too: building `temp` can throw, but that
  never touches the original `w`, and `swap` itself can't throw.
- "Mutate, *then* call something risky" (`f2`, `f4`, `f7`) leaves a changed
  but still-valid object if the risky call throws — `basic`.
- Chaining several only-`basic` operations (`f10`, via `operator=`) is at
  best `basic` overall.
- "Free the old resource, *then* try to acquire the new one, without
  clearing the handle in between" (`f5`, `f12`) leaves a dangling handle to
  already-freed memory if the acquisition throws — `none`.

## Example

For a hypothetical `f0(Widget& w) { w.active = true; }` (a plain field write
that can never fail): `classify_guarantees` should set the corresponding
entry to `0` (nothrow).

## What the gate checks

The driver reproduces all 12 functions verbatim and determines each one's
*actual* guarantee empirically: it runs the function once with no injected
failure to discover how many risky calls it makes, then re-runs it once per
risky call site with exactly that call forced to throw, observing the
resulting `Widget` state after every injected failure. A function is
`strong` only if the state is unchanged at **every** site, `basic` if it is
merely valid at every site, and `none` if any site leaves it invalid.

The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and
requires

$$ \mathrm{exact\_match} = 1.0 $$

against the reference across all 12 predictions.
