## Context

C++17 partitions every expression into one of three **value categories**:

| Category | Identity? | Can move from? | Typical examples |
|----------|-----------|----------------|------------------|
| **lvalue** | yes | no | named variables, dereferences, assignment results |
| **xvalue** (expiring) | yes | yes | `std::move(x)`, `static_cast<T&&>(x)` |
| **prvalue** (pure rvalue) | no | — | literals, arithmetic results, temporaries |

Formally, an expression is a **glvalue** if its evaluation determines the
identity of an object, bit-field, or function; it is a **prvalue**
otherwise. Among glvalues, an **xvalue** denotes an object about to be
moved from; the remaining glvalues are **lvalues**.

There is a real, mechanical test for this, not just memorized rules:
`decltype((E))` — note the **double parentheses**, which is significant —
reports `T&` if `E` is an lvalue, `T&&` if `E` is an xvalue, and plain `T`
(no reference) if `E` is a prvalue. `std::is_lvalue_reference` /
`std::is_rvalue_reference` on that type then gives you the category
directly, no guessing required.

## Task

Implement, in `solve.cpp`,

```cpp
std::vector<std::string> classify_value_categories();
```

Return, in order, one of `"lvalue"`, `"xvalue"`, or `"prvalue"` for each
of these 15 fixed expressions (`int x = 1;` is in scope wherever `x`
appears):

```
 0. x
 1. 42
 2. std::move(x)
 3. "hello"
 4. std::string("tmp")
 5. *(&x)
 6. x + 1
 7. ++x
 8. x++
 9. static_cast<int&&>(x)
10. std::declval<int&>()
11. std::declval<int&&>()
12. std::string("a") + std::string("b")
13. std::move(*(&x))
14. (x = 1)
```

## Example

`x` is a named variable → `"lvalue"`. `std::move(x)` casts to an rvalue
reference and the call expression is an **xvalue**, not an lvalue and not
a prvalue → `"xvalue"`. `42` has no identity → `"prvalue"`.

Two easy-to-miss ones: `"hello"` is a string literal with static storage
duration, so it is an **lvalue** (you can take its address) even though
it looks like a plain literal. `std::declval<int&>()` has return type
`int&` (reference collapsing on `add_rvalue_reference_t<int&>`), and a
function returning a non-rvalue reference produces an **lvalue** call
expression — not an xvalue, despite the name "declval".

## What the gate checks

The fixed driver (`main.cpp`) calls `classify_value_categories()` and
prints the 15 labels, one per line. The gate is an exact string match
(`exact_match == 1.0`) against the reference's printed output — the
reference determines every category with the real
`decltype((E))`/`is_lvalue_reference`/`is_rvalue_reference` mechanism
against the real compiler, not a memorized table, so every one of the 15
must genuinely match what the language says.
