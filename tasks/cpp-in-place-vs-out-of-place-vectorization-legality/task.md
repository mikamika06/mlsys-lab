## Context

Auto-vectorization of an elementwise loop over an array of records requires
determining whether vector execution can change the program's output because
of memory aliasing. When vectorizing with SIMD width $V$, the compiler
processes elements in blocks: for block $b$, it gathers all $V$ source values
into a vector register *before* scattering any of the $V$ computed
destination values back to memory.

Vectorization without a runtime pointer-overlap check is legal if and only
if:
1. The pointers are non-aliasing (e.g. declared `restrict`).
2. The access is exactly in-place ($\mathrm{src\_base} = \mathrm{dest\_base}$
   and $\mathrm{src\_size} = \mathrm{dest\_size}$) — reading and writing the
   very same bytes is always fine.
3. Otherwise: no loop-carried Read-After-Write hazard exists. For every
   vector block $b$, the bytes block $b$ **writes** must not overlap the
   bytes any **later** block $b' > b$ **reads**:
$$\left(\bigcup_{i=b \cdot V}^{(b+1)V - 1} \mathrm{WriteBytes}(i)\right) \cap \left(\bigcup_{j=(b+1)V}^{N-1} \mathrm{ReadBytes}(j)\right) = \emptyset$$

$N = 16$ array elements throughout. A source/dest access for element $i$ is
the byte range $[i \cdot \mathrm{struct\_size} + \mathrm{base},\ i \cdot
\mathrm{struct\_size} + \mathrm{base} + \mathrm{size})$, where
$\mathrm{base} = \mathrm{field\_offset} + \mathrm{elem\_shift} \times
\mathrm{field\_size}$ — the `elem_shift` models a stencil-style access that
reads or writes a neighboring element's field instead of its own (e.g.
`dest[i+1] = src[i]`).

## Task

Implement

```cpp
bool is_safe_to_vectorize(const KernelSpec& spec);
```

`KernelSpec` (declared in `sol.hpp`) carries every fact you need as plain
integers — `struct_size`, `src_offset`/`src_size`, `dest_offset`/
`dest_size`, `src_elem_shift`/`dest_elem_shift`, `has_restrict`,
`vector_width` — all computed by `main.cpp` with the real `sizeof`/
`offsetof` on real C++ structs, never a hand-rolled ABI table.

## Example

For a `struct { int f0; double f1; }` record, an in-place `int` access
(`src_offset == dest_offset`, `src_size == dest_size`) is always safe:

```cpp
KernelSpec s{16, 0, 4, 0, 4, 0, 0, false, 4};
is_safe_to_vectorize(s);  // true -- in-place exemption
```

## What the gate checks

`main.cpp` builds 10 real C++ structs matching 10 scenarios (in-place
access, `restrict`-qualified access, disjoint fields, a forward
element-shift, a backward element-shift, a differently-sized vector width,
and more), reads their true layout with `sizeof`/`offsetof`, and prints your
verdict for each. A shift of exactly $+1$ element with $V = 4$ is the one
case that actually creates a cross-block hazard here: the last lane of
block $b$ writes into the first lane's territory of block $b+1$, so that
scenario must come out **unsafe** — always answering "safe" fails it (and
the compiled reference's real numbers confirm the hazard, not a guess).
The grader compiles your `.cpp` with the real local `clang++` and requires
all 10 printed verdicts to match the reference's exactly
($\mathrm{exact\_match}=1.0$).
