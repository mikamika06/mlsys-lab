## Context

In C++, the **strict aliasing rule** lets a compiler assume two pointers do
not point at the same object (do not *alias*) when they have unrelated
types, which permits aggressive reordering and caching of loads/stores.
The compiler must instead be conservative — assume the pointers **may**
alias — when:

1. The types are the same, ignoring `const`/`volatile`/`signed`/`unsigned`
   qualifiers.
2. One type is a base class of the other, anywhere in the inheritance
   chain.
3. Either type is a character type (`char`, `signed char`, `unsigned
   char`) or `std::byte`, which the standard explicitly permits to alias
   any object.

Casting pointers in a way that violates these rules and then using them
as if they don't alias is Undefined Behavior.

## Task

Implement, in `solve.cpp`,

```cpp
int may_assume_no_alias(const std::string& type_a, const std::string& type_b,
                         const std::vector<std::pair<std::string, std::string>>& hierarchy);
```

`type_a` / `type_b` are type spellings such as `"int"`, `"unsigned int"`,
`"const float"`, `"char"`, `"std::byte"`, or a class name like `"Derived"`.
`hierarchy` is a list of `(derived, base)` pairs describing single
inheritance; a class with no base simply has an empty-string base (or does
not appear as a `derived` entry).

Algorithm:

1. Strip any leading `"const "`, `"volatile "`, `"unsigned "`, `"signed "`
   from both `type_a` and `type_b` (there may be more than one qualifier
   to strip, e.g. `"const unsigned int"`).
2. If either stripped type is `"char"` or `"std::byte"`, return `0`
   (may alias).
3. If the stripped types are identical, return `0`.
4. If one stripped type is a base of the other anywhere in `hierarchy`
   (walk the chain in either direction), return `0`.
5. Otherwise return `1` (the compiler may assume no-alias).

## Example

With `hierarchy = {("Derived", "Base")}`:

- `may_assume_no_alias("int", "float", hierarchy)` -> `1`
- `may_assume_no_alias("int", "unsigned int", hierarchy)` -> `0`
- `may_assume_no_alias("float", "char", hierarchy)` -> `0`
- `may_assume_no_alias("Derived", "Base", hierarchy)` -> `0`

## What the gate checks

The fixed driver (`main.cpp`) builds one hierarchy (`Derived`/`Derived2` ->
`Base`, `Unrelated` standalone) and runs 15 fixed pointer-type pairs
covering identical types, sign variants, cv-qualifier variants, char/byte
exceptions, base/derived pairs in both directions, and unrelated classes,
then prints the 15 `0`/`1` results as one line. The gate is an exact string
match (`exact_match == 1.0`) against the reference's printed line — every
one of the 15 classifications must be correct, not just most of them.
