## Context

A C++ program is built from many **translation units** (TUs) — each `.cpp` file
plus everything it `#include`s, compiled separately and then linked. The **One-Definition
Rule** (ODR, `[basic.def.odr]`) constrains how many times an entity may be *defined*:

- Within a single TU there may be at most **one** definition of any given entity.
- Across the whole program there must be **exactly one** definition of every
  non-inline function or variable that has *external linkage*.

The ODR then carves out a set of constructs that are explicitly *allowed* to be
defined in **more than one** TU, provided every definition is token-for-token
identical:

$$\text{multi-TU OK} \;=\; \{\text{class/union types},\; \text{enums},\; \text{templates},\; \text{inline functions},\; \text{inline variables}\}.$$

Two further facts complete the picture:

- An entity with **internal linkage** (a `static` function, an anonymous-namespace
  entity, or a `const` variable at namespace scope) — or with **no linkage** — is a
  *distinct* entity in each TU, so defining it in every TU is fine.
- A **type alias** (`typedef` / `using`) does not define an entity at all, so it too
  may appear in every TU.

Everything else — a non-inline free function, an out-of-line member function, a plain
non-`const` global variable — has external linkage and must be defined exactly once,
which is why those definitions live in a `.cpp` file and only their *declarations* go
in a header.

## Task

Implement in C++:

```cpp
int may_appear_in_multiple_tus(int kind, int linkage, int is_inline);
```

Each construct is described by three integers (see `sol.hpp` for the named enum
constants):

- `kind`: `KIND_FUNCTION`, `KIND_VARIABLE`, `KIND_CLASS`, `KIND_ENUM`, `KIND_ALIAS`, `KIND_TEMPLATE`
- `linkage`: `LINK_EXTERNAL`, `LINK_INTERNAL`, `LINK_NONE`
- `is_inline`: `1` if declared `inline` (or implicitly inline, e.g. `constexpr`), else `0`

Return `1` if a construct with these properties **may be defined in more than one
translation unit** without violating the ODR, else `0`.

Edit only `solve.cpp`. The driver `main.cpp` and the contract `sol.hpp` are fixed.

## Example

```
// struct Point { int x, y; };           kind=CLASS,    external, not inline
may_appear_in_multiple_tus(KIND_CLASS,    LINK_EXTERNAL, 0)  -> 1

// inline int sq(int) { ... }            kind=FUNCTION, external, inline
may_appear_in_multiple_tus(KIND_FUNCTION, LINK_EXTERNAL, 1)  -> 1

// static int helper() { ... }           kind=FUNCTION, internal
may_appear_in_multiple_tus(KIND_FUNCTION, LINK_INTERNAL, 0)  -> 1

// int add(int, int) { ... }             kind=FUNCTION, external, not inline
may_appear_in_multiple_tus(KIND_FUNCTION, LINK_EXTERNAL, 0)  -> 0

// int g_counter;                        kind=VARIABLE, external, not inline
may_appear_in_multiple_tus(KIND_VARIABLE, LINK_EXTERNAL, 0)  -> 0
```

## What the gate checks

The driver classifies 12 fixed constructs (free function, inline function, struct,
function template, global variable, inline variable, `const` namespace-scope variable,
`static` function, enum, type alias, class template, out-of-line member function),
prints each `0`/`1` verdict, then a packed 12-bit value and the count of "may appear
in multiple TUs" constructs. Your program's full output must match the reference
byte-for-byte (`exact_match == 1.0`).
