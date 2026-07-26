---
title: "What is move semantics?"
description: "Move semantics explained, with a copy-vs-move count across seven real operations run through clang++, including the noexcept switch that turns vector growth from moves into copies."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is move semantics?

Move semantics is C++'s mechanism for transferring a resource — a heap buffer, a file handle —
from a source object to a destination one by copying a handful of small fields instead of the
underlying data, whenever the compiler can prove the source is a temporary or has been told it
may be discarded. Below, one instrumented type run through clang++ counts exactly how many
copies and moves seven common operations really perform.

## Rvalue references and `std::move`

Every C++ expression has a *value category*. An **lvalue** names something with a persistent
address — a variable, a dereferenced pointer. An **rvalue** is a temporary with no name of its
own: the result of a function call, `a + b`, a braced literal. `T&&` is an **rvalue reference** —
a reference that can only bind to rvalues — and its entire purpose is letting overload
resolution tell "this argument is a throwaway, plunder it" apart from "this argument might be
read again, leave it alone." `std::move` does not move anything by itself; it is
`static_cast<T&&>(x)`, a compile-time relabeling that makes a named lvalue eligible to bind to
the rvalue-reference overload. The actual transfer happens inside whichever move constructor or
move assignment operator overload resolution then selects — and only if one exists. Without a
user-provided move constructor, `std::move` just steers the call to the copy constructor
instead: a label change with no cost saved, and a common source of the belief that "I moved it"
guarantees anything at all.

## Smart pointers vs raw pointers

`std::unique_ptr<T>` is move semantics turned into a design, not bolted onto one: its copy
constructor is `=delete`d, so two owners of the same resource are impossible by construction,
and its move constructor does three words of work — copy the pointer, null the source — which is
why returning one by value from a factory, or storing one in a growing `std::vector`, costs
nothing beyond that. A raw `T*` has no constructors for the compiler to select between, so
"moving" one is indistinguishable from copying it: the ambiguity over who is responsible for the
eventual `delete` is precisely what a smart pointer exists to remove. Deciding
[which smart pointer fits a given ownership scenario](../tasks/cpp-pick-the-right-smart-pointer-12-scenarios/task.md)
is a question about who owns the resource and for how long, never about which pointer type is
fastest to type.

## How it works

A class gets up to five special member functions: destructor, copy constructor, copy
assignment, move constructor, move assignment — the Rule of Five. The compiler generates the
move pair only when none of the other four has been user-declared; write a destructor to free a
resource and forget to also write a move constructor, and every move-looking call silently falls
back to copying, with no warning, because copying is always a legal — merely slower — substitute
for a move. [A `Buffer` with a hand-written, correct move constructor and move
assignment](../tasks/cpp-buffer-with-correct-move-ctor-assign/task.md) is that rule written out
in full, self-assignment included, since a plain "steal the pointer" breaks the moment the
source and destination alias.

The place this stops being a style preference is `std::vector`. Growing past capacity means
relocating every existing element into a new allocation, and the standard requires the vector to
call `std::move_if_noexcept` on each one: if the element's move constructor is declared
`noexcept`, the vector moves; if the move constructor exists but is merely allowed to throw, the
vector copies instead, to preserve the guarantee that a `push_back` which throws never leaves the
vector half-relocated. A move constructor that forgets to null the source after stealing its
pointer — [a bug `std::vector`'s reallocation exposes as a double
free](../tasks/cpp-move-ctor-forgets-to-null-double-free-fix/task.md) rather than a quiet
leak — is the other classic way this rule bites.

## Copies and moves, measured across seven operations

Every row below is the same instrumented type, `Probe`, whose copy constructor deep-copies a
small heap array and whose move constructor steals the pointer and nulls the source; a global
counter increments inside whichever one the compiler actually calls. The last two rows are the
identical loop — eight `push_back` calls into an empty vector — with only the `noexcept` on the
move constructor changed.

| operation | copies | moves | final capacity |
|---|---|---|---|
| pass by value (lvalue argument) | 1 | 0 | — |
| pass by const reference | 0 | 0 | — |
| return a local (NRVO) | 0 | 0 | — |
| `push_back` an lvalue | 1 | 0 | — |
| `push_back(std::move(lvalue))` | 0 | 1 | — |
| 8 pushes from empty, move ctor `noexcept` | 0 | 15 | 8 |
| 8 pushes from empty, move ctor **not** `noexcept` | 7 | 8 | 8 |

Reproduce it — compiles and runs the exact type above with the local `clang++`:

```bash
python3 - <<'PY'
import os
import subprocess
import tempfile

SRC = r"""
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

static long g_copies = 0;
static long g_moves = 0;

template <bool NoexceptMove>
struct Probe {
    double* data;
    int n;

    explicit Probe(int n_) : data(new double[n_]), n(n_) {
        for (int i = 0; i < n; ++i) data[i] = i;
    }
    Probe(const Probe& o) : data(new double[o.n]), n(o.n) {
        std::memcpy(data, o.data, n * sizeof(double));
        ++g_copies;
    }
    Probe(Probe&& o) noexcept(NoexceptMove) : data(o.data), n(o.n) {
        o.data = nullptr; o.n = 0;
        ++g_moves;
    }
    Probe& operator=(const Probe&) = delete;
    Probe& operator=(Probe&&) = delete;
    ~Probe() { delete[] data; }
};

using P = Probe<true>;
using PCopy = Probe<false>;

void by_value(P p) { (void)p; }
void by_const_ref(const P& p) { (void)p; }

P make_local() {
    P local(3);
    return local;
}

static void reset() { g_copies = 0; g_moves = 0; }
static void report(const char* label) {
    std::printf("%-24s copies=%ld moves=%ld\n", label, g_copies, g_moves);
}
static void report_growth(const char* label, size_t cap) {
    std::printf("%-24s copies=%ld moves=%ld final_capacity=%zu\n", label, g_copies, g_moves, cap);
}

int main() {
    { P p(3); reset(); by_value(p); report("pass_by_value"); }
    { P p(3); reset(); by_const_ref(p); report("pass_by_const_ref"); }
    { reset(); P r = make_local(); report("return_local_nrvo"); (void)r; }
    {
        std::vector<P> v; v.reserve(4);
        P p(3);
        reset();
        v.push_back(p);
        report("push_back_lvalue");
    }
    {
        std::vector<P> v; v.reserve(4);
        P p(3);
        reset();
        v.push_back(std::move(p));
        report("push_back_moved");
    }
    {
        std::vector<P> v;
        reset();
        for (int i = 0; i < 8; ++i) v.push_back(P(3));
        report_growth("growth_noexcept_move", v.capacity());
    }
    {
        std::vector<PCopy> v;
        reset();
        for (int i = 0; i < 8; ++i) v.push_back(PCopy(3));
        report_growth("growth_throwing_move", v.capacity());
    }
    return 0;
}
"""

with tempfile.TemporaryDirectory() as d:
    src_path = os.path.join(d, "probe.cpp")
    exe_path = os.path.join(d, "probe")
    with open(src_path, "w") as f:
        f.write(SRC)
    subprocess.run(["clang++", "-O2", "-std=c++20", "-o", exe_path, src_path], check=True)
    out = subprocess.run([exe_path], capture_output=True, text=True, check=True).stdout
print(out)
PY
```

The first five rows are unsurprising once value categories are separated from mechanism: a
named lvalue always copies unless explicitly `std::move`'d, and a reference parameter never
constructs anything. The last two rows are the one worth remembering. Both loops run the same
eight `push_back` calls against the same doubling growth policy — capacity climbs `0→1→2→4→8`,
relocating `0+1+2+4=7` old elements across four reallocations, plus one insertion move per push —
and the *only* difference between `moves=15, copies=0` and `moves=8, copies=7` is one keyword on
the move constructor's declaration.

## Practise it

```bash
mlsys grade cpp-noexcept-move-unlocks-vector-growth-moves
```

[That task](../tasks/cpp-noexcept-move-unlocks-vector-growth-moves/task.md) hands you the same
switch — a `bool move_is_noexcept` parameter controlling whether `Elem`'s move constructor is
declared `noexcept(true)` or `noexcept(false)` — across seven fixed `(element_size, n_pushes,
move_is_noexcept)` cases, and gates on an exact byte-for-byte match of every printed
`GrowthCounts` field against a reference built from a real `std::vector`. Nothing here is
simulated: get the growth arithmetic right but leave the `noexcept` off the move constructor and
every count for the `true` cases quietly turns into the `false` cases' numbers instead, with no
compiler warning to catch it.

In increasing difficulty:
[classify a value category from a bare expression](../tasks/cpp-value-category-of-15-expressions/task.md),
[count copies across value, const-ref and ref parameters](../tasks/cpp-copy-count-across-value-const-ref-ref-params/task.md),
[classify twelve returns as rvo, nrvo, move or copy](../tasks/cpp-classify-12-returns-nrvo-move-copy/task.md),
[reproduce the exact capacity sequence a growth policy produces](../tasks/cpp-growth-policy-reproduce-the-capacity-sequence/task.md),
and [count reallocations and element moves across a full push_back run](../tasks/cpp-count-reallocations-element-moves-on-push-back/task.md).

## Common mistakes

- **Declaring the move constructor without `noexcept`.** As measured above, this is not a style
  nit — it silently converts every `std::vector` reallocation from zero-cost moves into full
  deep copies, and nothing in the build fails to tell you.
- **Forgetting to null the source inside the move constructor.** Stealing `other.ptr` without
  setting it back to null leaves two live objects owning the same resource; the moment either
  destructor runs, the other's pointer is dangling, and [`std::vector`'s reallocation is what
  usually finds it](../tasks/cpp-move-ctor-forgets-to-null-double-free-fix/task.md), as a crash
  far from the constructor that caused it.
- **Writing `return std::move(local);`.** `std::move` turns a plain local-variable return
  expression into something that is no longer eligible for NRVO, forcing a real move where the
  compiler would otherwise have elided construction entirely —
  [removing that `std::move` is the whole fix](../tasks/cpp-remove-return-std-move-pessimization/task.md).
- **Assuming `std::move` on a `const` object does anything.** It produces a `const T&&`, which
  binds to the copy constructor's `const T&` parameter just as well as an lvalue does, so the
  "move" silently copies — the type system allows it precisely because a copy is always a legal,
  if wasteful, substitute for a move.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md):

- **[LearnCpp.com](https://www.learncpp.com/)** — a free, complete chapter on move semantics and
  smart pointers, each lesson ending in a self-check quiz with the answer hidden until clicked.
  Explains the same rules; nothing runs your code or counts anything for you.
- **[Stanford CS106L assignments](https://github.com/cs106l/cs106l-assignments)** — assignment 6
  is move semantics directly, assignment 7 has you implement your own `unique_ptr`, both with a
  real local autograder. The closest cousin to this page's approach, minus the instrumented
  counting.
- **[CppQuiz.org](https://cppquiz.org/)** — 190 short snippets you predict the exact output of,
  including several on move construction and copy elision, scored the instant you submit.
- **[Exercism's C++ track](https://exercism.org/tracks/cpp)** — general-purpose exercises with a
  real automated test suite per exercise plus optional human mentor review; not move-semantics
  specific, but free and genuinely graded.
- **cppreference**, references 1–3 below, remains the primary source for the exact rules.
  Read it after measuring, not instead.

## References

1. cppreference, *Move constructors*.
   https://en.cppreference.com/w/cpp/language/move_constructor
2. cppreference, *Value categories*.
   https://en.cppreference.com/w/cpp/language/value_category
3. cppreference, `std::move`.
   https://en.cppreference.com/w/cpp/utility/move
4. C++ Core Guidelines, *C.66: Make move operations noexcept*.
   https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c66-make-move-operations-noexcept
