## Context

When building a shared library (`.so` / `.dylib`), the compiler and linker
decide which symbols (functions, global variables) get **exported** — made
visible to other libraries and executables that link against it — and which
stay private. Exporting too many symbols bloats the library and slows down
dynamic linking, so a common optimization is `-fvisibility=hidden`: it
flips the *default* visibility of every symbol to hidden, and the developer
then explicitly tags public API functions with
`__attribute__((visibility("default")))`.

Visibility attributes interact with linkage in three layered rules:

1. `static` functions have **internal linkage** — they are private to their
   translation unit and are **never** exported, no matter what visibility
   attribute is also written on them.
2. An **explicit** `__attribute__((visibility(...)))` on an externally
   linked function always overrides the global `-fvisibility` flag.
3. With **no** explicit attribute, an externally linked function inherits
   the global default: exported when the library was *not* built with
   `-fvisibility=hidden`, hidden when it was.

## Task

Implement

```cpp
int count_exported_symbols(bool global_hidden, const int* is_static, const int* attr, int n);
```

For `n` declarations, `is_static[i]` is `1` if declaration `i` is `static`,
and `attr[i]` is its explicit visibility: `0` = `visibility("default")`,
`1` = `visibility("hidden")`, `2` = no explicit attribute. Apply the three
rules above and return how many of the `n` declarations end up exported.

## Example

```
global_hidden = true
decls: [ {static=false, attr=none}, {static=true, attr=default},
         {static=false, attr=hidden}, {static=false, attr=default} ]

[0] no explicit attr, inherits global (hidden)   -> not exported
[1] static -> internal linkage, attr irrelevant  -> not exported
[2] explicit hidden                              -> not exported
[3] explicit default                             -> exported
count_exported_symbols(...) == 1
```

## What the gate checks

For each of 5 fixed scenarios, the driver does not just check your
prediction against a rule engine — it generates a REAL `.cpp` file
reproducing the exact declarations (real `static`, real
`__attribute__((visibility(...)))`), compiles it with the real `clang++`
into a real `.dylib` (honoring `-fvisibility=hidden` per scenario), and
counts the real exported dynamic symbols with `nm -gU` (extern-only,
defined-only). That is the ground truth your prediction is checked against.

The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and
requires

$$ \mathrm{exact\_match} = 1.0 $$

against the reference across all 5 scenarios — including the case where
`global_hidden` is false (so an unattributed declaration DOES get exported)
and the all-`static` case (nothing is ever exported, attributes or not).
