#pragma once

// The measured object-layout signature of ONE version of a struct/class.
// The driver fills this in with FACTS obtained from the real compiler
// (sizeof, alignof, std::is_polymorphic, and the byte offset of each field
// that exists in both versions).
struct Layout {
    int size;      // sizeof(T)   in bytes
    int align;     // alignof(T)  in bytes
    int vptr;      // 1 if the type is polymorphic (carries a vptr), else 0
    int nfields;   // number of fields common to BOTH versions
    int off[16];   // off[i] = byte offset of the i-th common field, same order in both versions
};

// Classify ONE struct edit (old version -> new version).
// Return 1 if the edit is ABI-breaking, i.e. it changes the in-memory OBJECT
// layout that already-compiled code depends on; return 0 if the object layout
// is unchanged (ABI-compatible).
//
// TODO(learner): implement this in solve.cpp.
int abi_breaks(const Layout& old_v, const Layout& new_v);
