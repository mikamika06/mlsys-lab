#pragma once

// Widget has two overloads of read(), distinguished ONLY by the const
// qualification of the implicit object parameter:
//   int read();          -- callable on a mutable Widget: MUST mutate
//                            `calls` (increment it) and return the new value.
//   int read() const;    -- callable on a const Widget: MUST NOT mutate
//                            `calls`, and must return it unchanged.
//
// Under the real Itanium C++ ABI (what clang++ actually emits on macOS),
// these compile to two entirely separate linker symbols -- the const
// overload's mangled name carries a 'K' qualifier that the non-const
// overload's does not (_ZN6Widget4readEv vs _ZNK6Widget4readEv). main.cpp
// proves both symbols genuinely exist by reading this executable's own
// symbol table with `nm` at runtime.
//
// `calls` is declared `mutable`, which means the COMPILER will happily let
// a const member function write to it -- `const` on a member function only
// promises "I won't reassign non-mutable members", not "nothing changes".
// The read() const overload must still behave as a real read-only view: it
// must not use that loophole to mutate `calls`, even though it legally could.
struct Widget {
    mutable int calls;

    int read();          // non-const: must mutate `calls`
    int read() const;    // const: must NOT mutate `calls`, despite being able to
};
