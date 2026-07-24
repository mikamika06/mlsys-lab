## Context

In C++, **internal linkage** means a name is only visible within its translation unit (TU). There are two mechanisms to achieve this at namespace scope:

1. **`static` keyword** — marks a variable or function as having internal linkage.
2. **Anonymous (unnamed) namespace** — every declaration inside `namespace { ... }` automatically gets internal linkage.

Both prevent the linker from seeing the symbol outside the TU. However, they differ in subtle but important ways:

- **`static` only applies to functions and variables.** It cannot be written in front of a class/struct definition, a template, an enum, or a typedef/using alias — those declarations are simply not `static`-able, so there is no way to give them internal linkage with the keyword.
- **Anonymous namespaces work uniformly** for functions, variables, types, and templates — anything declared inside one gets internal linkage, no exceptions.
- **`const` at namespace scope already has internal linkage by default** in C++ (unlike C). Adding `static` is redundant but harmless, and matches what an anonymous namespace would also give it — the two are equivalent for this category.
- **`inline` variables (C++17)** are subtly different: `static inline int x = 1;` still names one TU-local definition, the usual `static` story. Moving the same declaration into an anonymous namespace instead makes it a distinct internal-linkage entity *per TU* with its own address — a real behavioral difference from the `static` version, not just a stylistic one.
- **`extern` and `static` are contradictory** on the same declaration (`extern` asks for external linkage, `static` forces internal — not something you write together). Putting an `extern` declaration inside an anonymous namespace still ends up with internal linkage, but that is a different declaration shape than "extern with static", so the two are not considered equivalent forms here.

The question for each category: does writing the declaration with `static` produce the **same observable linkage behavior** as writing the same declaration inside an anonymous namespace instead — both legal, both internal linkage, no ODR difference? Or do the two diverge?

## Task

Implement

```cpp
int is_equivalent(Category c);
```

where `Category` is one of:

```cpp
enum Category {
    FREE_FUNCTION, FREE_VARIABLE, CONST_VARIABLE, CLASS_TYPE,
    FUNCTION_TEMPLATE, CLASS_TEMPLATE, INLINE_VARIABLE, ENUM_TYPE,
    TYPEDEF_ALIAS, EXTERN_VARIABLE,
};
```

Return `1` if `static` and an anonymous namespace give equivalent internal-linkage behavior for that category, `0` if they differ (because `static` cannot legally be applied to it, or because the resulting semantics diverge as described above).

The driver calls `is_equivalent` on all 10 categories, in the fixed order shown above, and prints one bit per category followed by the whole answer packed into a single integer and its popcount.

## Example

For the first four categories:

```
FREE_FUNCTION  -> 1   (static void f(){} behaves like namespace{ void f(){} })
FREE_VARIABLE  -> 1   (static int x; behaves like namespace{ int x; })
CONST_VARIABLE -> 1   (const already implies internal linkage; static is redundant but matches)
CLASS_TYPE     -> 0   (static class C {}; is ill-formed; anonymous namespace is the only legal form)
```

## What the gate checks

The driver prints the 10-bit vector, `packed=<int>` and `count=<int>`. The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and compares every printed number against `ref.cpp`:

$$
\mathrm{exact\_match} = 1 \iff \text{every printed token (bits, packed value, count) matches the reference byte-for-byte}
$$

A classifier that always guesses "equivalent" (or always "diverges") gets some categories right by luck but is wrong on at least `FREE_FUNCTION`/`FREE_VARIABLE`/`CONST_VARIABLE` vs. the rest, so the packed integer and count both diverge from the reference — the gate requires getting **all ten** categories right, not most of them.
