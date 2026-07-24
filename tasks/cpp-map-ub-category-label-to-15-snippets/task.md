## Context

Undefined Behavior (UB) lets the compiler assume certain conditions never
happen, and optimize aggressively on that assumption. When code violates
one of those assumptions, the "optimization" turns into a bug. Different
sanitizers (ASan, UBSan, TSan) catch different UB categories:

- `lifetime`: use-after-free, returning references to locals, dangling
  references into a container that reallocated.
- `aliasing`: strict-aliasing violations (type punning via pointer casts).
- `integer`: signed integer overflow, shifts by an amount $\ge$ the
  bit-width.
- `bounds`: out-of-bounds array access.
- `data-race`: unsynchronized concurrent access to the same memory (at
  least one side a write).
- `sequencing`: unsequenced modifications of a scalar within one
  expression (e.g. `i++ + i++`).
- `null`: dereferencing a null pointer.

## Task

Implement `classify_ub` in `solve.cpp`:

```cpp
const char* classify_ub(const char* snippet);
```

`snippet` is a C string of C++ source text known to contain exactly one UB
instance. Classify it using substring search (e.g. `strstr`), checking **in
this exact order** — first match wins:

1. contains `"thread"` &rarr; `"data-race"`
2. contains `"<<"` or `"2147483647"` &rarr; `"integer"`
3. contains `"return &"`, `"delete"`, or `"push_back"` &rarr; `"lifetime"`
4. contains `"(float*)"` or `"(short*)"` &rarr; `"aliasing"`
5. contains `"nullptr"` or `"*p = 5"` &rarr; `"null"`
6. contains `"i++"` &rarr; `"sequencing"`
7. contains `"arr["` &rarr; `"bounds"`
8. otherwise &rarr; `"unknown"`

The order matters: e.g. `"arr[i] = i++;"` contains both `"i++"` and
`"arr["`, but rule 6 fires first, so it's `"sequencing"` (the classic
unsequenced read/write bug), not `"bounds"`.

The fixed driver in `main.cpp` runs your function over 15 fixed canonical
snippets and prints the 15 labels.

## Example

```cpp
classify_ub("int* f() { int x = 5; return &x; }");     // -> "lifetime"
classify_ub("int f(int i) { return i++ + i++; }");      // -> "sequencing"
```

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with real
`clang++ -O2 -std=c++20`, runs it, and compares stdout byte-for-byte against
the reference build (`exact_match == 1.0`) across all 15 snippets — two or
three examples of each of the seven categories. The starter labels
everything `"unknown"`, which is wrong for all 15.
