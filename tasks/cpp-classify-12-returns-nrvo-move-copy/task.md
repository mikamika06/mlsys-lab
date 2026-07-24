## Context

In C++17, returning an object by value can result in different behaviors depending on what is returned:
- **`rvo`**: Return Value Optimization (C++17 guaranteed copy elision for prvalues). No copy or move constructor is called.
- **`nrvo`**: Named Return Value Optimization. The compiler is permitted to elide the copy/move if you return a local variable by name. (If elision doesn't happen, it falls back to a move).
- **`move`**: The object is implicitly or explicitly moved (e.g., returning a by-value parameter or using `std::move`).
- **`copy`**: The object must be copied (e.g., returning an lvalue reference, a global, or a subobject).

Consider the following struct `T` and 12 return scenarios:
```cpp
struct T {
    char c;
    double d;
    int i;
};

// Assume T has accessible copy/move constructors.

// 1.
T f1() { T t; return t; }

// 2.
T f2() { return T(); }

// 3.
T f3(T t) { return t; }

// 4.
T f4(T& t) { return t; }

// 5.
T f5() { T t; return std::move(t); }

// 6.
T g;
T f6() { return g; }

// 7.
T f7() { static T t; return t; }

// 8.
T f8() { T* t = new T(); return *t; }

// 9.
T f9(bool b) { T t1, t2; return b ? t1 : t2; }

// 10.
T f10() { T t; return (t); }

// 11.
struct U { T t; };
T f11(U u) { return u.t; }

// 12.
T f12() { return T{ 'a', 1.0, 42 }; }
```

## Task

Implement

```cpp
void predict_return_kinds(std::string out[12]);
int predict_struct_size();
```

- Write your classification (`"nrvo"`, `"rvo"`, `"move"`, or `"copy"`) for
  `f1` through `f12` into `out[0..12)` (`out[0]` is `f1`, `out[11]` is `f12`).
- Return your prediction for `sizeof(T)` in bytes under LP64.

## Example

If you believed `f1` elides via NRVO and `sizeof(T)` is 24:
`out[0] = "nrvo";` and `predict_struct_size()` returns `24`.

## What the gate checks

`main.cpp` is a fixed driver that defines the instrumented `struct T` (its
copy and move constructors each bump a global counter) and the 12 functions
`f1`..`f12` exactly as written above, and actually **runs** every one of
them. For each call it resets the counters, invokes the function, and reads
back how many copy/move constructor calls really happened:

- 0 copies and 0 moves -> elided (`"rvo"` if the return operand was a
  prvalue, `"nrvo"` if it named a single local automatic object),
- 0 copies and $\ge 1$ moves -> `"move"`,
- $\ge 1$ copies -> `"copy"`.

That gives a real, runtime-measured ground truth for every snippet — never
a hardcoded table. (Arguments to `f3` and `f11`, which take `T`/`U` by
value, are passed as prvalues so the mandatory elision of the *argument*
doesn't get mixed into the count for the *return*.)

The driver then compares your 12 predictions and your size prediction
against that ground truth, printing a line per snippet plus the match count
and the size check. The grader recompiles your `solve.cpp` against the same
fixed driver and requires the printed output to match the reference's
exactly ($\mathrm{exact\_match}=1.0$). A single guessed category ("always
copy", "always move") only accidentally matches a few snippets and fails
the rest — you have to reason about each source correctly: prvalue vs.
named local vs. reference vs. global vs. static vs. dereferenced pointer
vs. ternary of two locals vs. subobject access.
