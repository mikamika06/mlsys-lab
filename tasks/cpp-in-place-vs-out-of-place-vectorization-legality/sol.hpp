#pragma once
// One elementwise-kernel vectorization-legality query, expressed directly in
// terms of REAL struct layout facts -- main.cpp computes every offset/size
// field below with the real `sizeof`/`offsetof` on a real C++ struct (see
// task.md), never a hand-rolled ABI table.
struct KernelSpec {
    int struct_size;      // real sizeof(the record struct)
    int src_offset;        // real byte offset of the source field within the struct
    int src_size;           // real sizeof of the source field's type
    int dest_offset;         // real byte offset of the dest field within the struct
    int dest_size;            // real sizeof of the dest field's type
    int src_elem_shift;        // extra whole source-field-widths to shift the access by
    int dest_elem_shift;        // extra whole dest-field-widths to shift the access by
    bool has_restrict;           // true if the pointers are declared `restrict` (proven non-aliasing)
    int vector_width;             // SIMD vector width V
};

// The kernel logically runs `for (i = 0; i < N; i++) dest[i] = f(src[i]);`
// over N = 16 array elements of the record struct, where "dest[i]" and
// "src[i]" are byte ranges [i*struct_size + base, i*struct_size + base +
// size) with base = field_offset + elem_shift * field_size.
//
// Return true if the kernel can be vectorized in blocks of `vector_width`
// WITHOUT a runtime overlap check: for every vector block b, the bytes that
// block WRITES must not overlap the bytes any LATER block READS. See
// task.md for the full rule (including the `restrict` and pure-in-place
// exemptions).
bool is_safe_to_vectorize(const KernelSpec& spec);
