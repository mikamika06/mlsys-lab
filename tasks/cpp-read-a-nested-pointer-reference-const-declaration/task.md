## Context

C++ declarations that combine pointers, references, and `const` can be
confusing because qualifiers bind in non-obvious ways. Rather than parsing
the *text* of a declaration, this task has you classify the real *type* it
names, using the standard library's compile-time type traits:

- `std::is_lvalue_reference_v<T>` / `std::is_rvalue_reference_v<T>` — is T a
  `&` or `&&`?
- `std::is_pointer_v<T>` — is T a pointer (a plain `U*` or a const-qualified
  `U* const`)?
- `std::is_const_v<T>` — is T itself top-level `const`? (For `int* const`,
  the *pointer* is const; for `const int*`, the pointer isn't const, its
  *pointee* is.)
- `std::remove_reference_t<T>` / `std::remove_pointer_t<T>` — peel off one
  layer and get the type underneath.

- `int*` — pointer to `int`
- `const int*` — pointer to `const int` (the pointee is const, the pointer is not)
- `int* const` — const pointer to `int` (the pointer is const, the pointee is not)
- `int&` — lvalue reference to `int`
- `int&&` — rvalue reference to `int`

With multiple `*` levels the chain grows: `int**` is *pointer to pointer to
`int`*, `int* const*` is *pointer to (const pointer to `int`)*.

## Task

Implement

```cpp
template <typename T>
std::string classify_type();
```

Peel off one layer of `T` at a time (reference first, then pointer) and
recurse on what's left, building the label from the outside in:

**Label format rules**

| Type | Label |
|---|---|
| `int*` | `pointer-to-int` |
| `const int*` | `pointer-to-const-int` |
| `int* const` | `const-pointer-to-int` |
| `const int* const` | `const-pointer-to-const-int` |
| `int&` | `ref-to-int` |
| `const int&` | `ref-to-const-int` |
| `int&&` | `rvalue-ref-to-int` |
| `const int&&` | `rvalue-ref-to-const-int` |
| `int**` | `pointer-to-pointer-to-int` |
| `const int**` | `pointer-to-pointer-to-const-int` |
| `int* const*` | `pointer-to-const-pointer-to-int` |
| `int*&` | `ref-to-pointer-to-int` |

## Example

```cpp
classify_type<int*>();      // -> "pointer-to-int"
classify_type<const int*>(); // -> "pointer-to-const-int"
classify_type<int* const>(); // -> "const-pointer-to-int"
classify_type<int**>();      // -> "pointer-to-pointer-to-int"
classify_type<int*&>();      // -> "ref-to-pointer-to-int"
```

## What the gate checks

`run_declaration_tests()` (which you also implement, right below
`classify_type`) explicitly instantiates `classify_type<T>()` for the same
12 types shown in the table above and prints one `<index> <label>` line per
type. The grader compiles your `.cpp` with the real local `clang++` and
requires all 12 printed lines to match the reference's exactly
($\mathrm{exact\_match} = 1$). Getting the outer layer right but the inner
recursion wrong (e.g. treating `int* const*`'s inner pointer as non-const)
only gets you partial credit on that line — which still fails the whole
gate, since every line must match.
