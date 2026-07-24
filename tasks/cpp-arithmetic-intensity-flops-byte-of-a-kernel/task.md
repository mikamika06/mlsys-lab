## Context

The **arithmetic intensity** of a kernel is the ratio of floating-point operations (FLOPs) performed to the amount of memory traffic (in bytes) transferred between the cache and main memory. It is a critical metric for understanding whether a kernel is compute-bound or memory-bound.

When processing arrays of data, the data layout profoundly affects the memory traffic:
- **Array of Structs (AoS)**: A single array where each element is a struct containing all fields. Accessing even one field typically requires loading the entire struct (including padding) into the cache. If any field in the struct is written, the entire struct is eventually written back to memory.
- **Struct of Arrays (SoA)**: A struct where each field is stored in its own separate, contiguous array. This allows the kernel to load and store only the fields that are actually accessed, avoiding the overhead of transferring unneeded fields or padding.

This task uses **real structs compiled by clang++**, not a re-implemented ABI model: `main.cpp` defines a handful of structs with different padding profiles and reads their sizes with the actual `sizeof()` operator. Whatever padding the real compiler inserts is exactly what your function receives as `struct_bytes`.

## Task

Implement, in `solve.cpp`:

```cpp
double arithmetic_intensity(int struct_bytes,
                             const int* field_bytes, int num_fields,
                             const int* reads, int num_reads,
                             const int* writes, int num_writes,
                             int flops, bool is_aos);
```

- `struct_bytes`: `sizeof()` of the struct, as measured by the real compiler (padding included).
- `field_bytes[i]`: `sizeof()` of field `i`, in declaration order.
- `reads` / `num_reads`: indices into `field_bytes` that the kernel reads.
- `writes` / `num_writes`: indices into `field_bytes` that the kernel writes.
- `flops`: floating-point operations performed, per element.
- `is_aos`: `true` for Array-of-Structs layout, `false` for Struct-of-Arrays.

For `is_aos == true`: touching *any* field for reading pulls in the **whole struct** (`struct_bytes`) once; touching *any* field for writing pushes out the **whole struct** once. A kernel that both reads and writes moves `2 * struct_bytes` per element (a read pass and a separate write pass).

For `is_aos == false` (SoA): only the touched columns move. `bytes = sum(field_bytes[i] for i in reads) + sum(field_bytes[i] for i in writes)`.

Return `flops / bytes` as a `double`. If `bytes == 0`, return `+infinity` (`std::numeric_limits<double>::infinity()`).

## Example

```cpp
// struct { char a; int b; double c; };  -> real sizeof() = 16 (padding after char and after int)
// field_bytes = {1, 4, 8}
// read {0, 1}, write {2}, flops = 12, AoS
// -> loads struct (16 bytes) + stores struct (16 bytes) = 32 bytes -> 12 / 32 = 0.375

// same fields, SoA
// -> reads char (1) + int (4), writes double (8) = 13 bytes -> 12 / 13 ~= 0.9230769
```

## What the gate checks

`main.cpp` builds four real structs with different padding profiles (some with internal or tail padding, some with none), reads their real `sizeof()`s, and runs ten read/write/flops/layout scenarios through your function, printing each result. The reference (`ref.cpp`) is compiled and run the same way, and your printed numbers are compared against it: `max_abs_err <= 1e-9`. Conflating AoS with "sum of touched field bytes" (i.e. applying the SoA rule under an AoS layout) gives the wrong answer for every struct that actually has padding, which every scenario here is built to expose.
