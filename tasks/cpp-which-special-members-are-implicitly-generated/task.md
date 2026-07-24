## Context

C++ automatically generates six "special member functions" for a class —
default constructor, destructor, copy constructor, copy assignment
operator, move constructor, move assignment operator — but declaring one
of them by hand can silently suppress or delete the others. The rules (the
"Rule of Five", C++11 onward):

- **Default constructor**: not generated if the class declares any other
  constructor (a copy or move constructor counts; `= default`d or
  `= delete`d counts as declaring it).
- **Copy constructor / copy assignment**: each is implicitly **deleted**
  (not just absent — actively `= delete`d) if the class declares a move
  constructor or a move assignment operator.
- **Move constructor / move assignment**: each is generated only if the
  class declares **none** of: a destructor, a copy constructor, a copy
  assignment operator, or the *other* move member. Any of those present
  means the move member is simply not generated at all (unlike copy, it
  isn't deleted — it's just absent).
- **Destructor**: always available unless explicitly deleted (not
  exercised by this task's test classes).

## Task

Implement `classify_special_members` in `solve.cpp`:

```cpp
MemberAvail classify_special_members(const ClassDecl& d);
```

`ClassDecl` (declared in `sol.hpp`) describes which special members a class
explicitly declares. Apply the rules above and return which of the six end
up available (`true`) versus deleted/absent (`false`). See the detailed
per-member rules documented in `sol.hpp`.

The fixed driver in `main.cpp` runs your function over 10 fixed
`ClassDecl`s that mirror 10 real class definitions (documented as comments
in `main.cpp`) and prints the six-bit result for each.

## Example

```cpp
// struct C4 { C4(C4&&){} };   -- a user-declared move constructor only
ClassDecl d{0, false, false, false, /*move_ctor=*/true, false};
classify_special_members(d);
// -> default=false (a constructor was declared), dtor=true,
//    copy_ctor=false, copy_assign=false (both deleted by the move ctor),
//    move_ctor=true (user), move_assign=false (suppressed by the move ctor)
```

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with real
`clang++ -O2 -std=c++20`, runs it, and compares stdout byte-for-byte against
the reference build (`exact_match == 1.0`) across all 10 classes — covering
destructor-only, copy-only (ctor or assignment alone), move-only (ctor or
assignment alone), a user-provided default constructor, the classic
copy-ctor+copy-assign "rule of three", a defaulted-move-with-destructor
case, and an explicitly deleted default constructor. The starter reports
every member as available for every class, which is wrong for most of
them (e.g. it never reports the copy-deletion caused by a user move
member).
