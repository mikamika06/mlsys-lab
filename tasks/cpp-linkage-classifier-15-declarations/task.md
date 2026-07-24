## Context

In C++, every name introduced by a declaration has **linkage**, which
determines whether the same name elsewhere refers to the same entity:

- **External linkage**: the name can be referred to from other
  translation units (TUs).
- **Internal linkage**: the name can be referred to from anywhere in the
  current TU, but not from other TUs.
- **No linkage**: the name can only be referred to from the scope it was
  declared in.

Consider these 15 declarations, at namespace scope unless noted:

```cpp
 1.  int d1;
 2.  static int d2;
 3.  const int d3 = 5;
 4.  extern const int d4 = 5;
 5.  void d5();
 6.  static void d6();
 7.  inline void d7() {}
 8.  extern int d8;
 9.  class C { static int d9; };  // target: d9
10.  void f() { int d10; }        // target: d10
11.  void f() { static int d11; } // target: d11
12.  constexpr int d12 = 10;
13.  extern "C" void d13();
14.  const int* d14 = nullptr;
15.  int* const d15 = nullptr;
```

and this struct:

```cpp
struct S {
    double d;
    int    i;
};
```

## Task

Implement, in `solve.cpp`,

```cpp
std::pair<std::vector<std::string>, long> classify_linkage();
```

Return `{labels, struct_size}`:

- `labels`: 15 strings, `"external"`, `"internal"`, or `"none"`, for `d1`
  through `d15` in order.
- `struct_size`: `sizeof(S)` as the real compiler lays it out.

Key rules:

- `static` at namespace scope forces internal linkage.
- A non-`extern`, non-`inline` object at namespace scope whose own
  declared type is const-qualified defaults to internal linkage
  (`constexpr` implies `const`, so it gets the same default). Marking it
  `extern` overrides that default back to external.
- Watch the difference between "pointer to const" and "const pointer":
  `const int* d14` is a plain, non-const-qualified *pointer* (it just
  points at something const) — ordinary default linkage applies. `int*
  const d15` is itself const-qualified — the const-default-internal rule
  applies to it.
- A `static` data member of a class has external linkage by default (it
  needs exactly one definition across the whole program).
- A function-scope local variable has **no linkage**, whether or not it
  is itself declared `static` — only its storage duration changes, its
  name is still invisible outside the function.

## Example

`d2` (`static int d2;`) is internal — `static` at namespace scope always
is. `d10` (a local `int` inside `f()`) has no linkage at all — its name
means nothing outside `f`'s body, even though `d5` (an ordinary
namespace-scope function) has external linkage by default.

## What the gate checks

The fixed driver (`main.cpp`) calls `classify_linkage()` and prints
`struct_size` followed by the 15 labels, one per line. The gate is an
exact string match (`exact_match == 1.0`) against the reference's printed
output — every one of the 15 classifications must match the real C++
linkage rules, not just most of them.
