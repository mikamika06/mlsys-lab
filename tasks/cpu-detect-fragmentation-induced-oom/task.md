## Context

"Out of memory" doesn't always mean *out* of memory. An allocator can be
sitting on plenty of free bytes in total and still fail a request,
because none of its individual free blocks is big enough on its own.
This is **external fragmentation**: repeated alloc/free cycles chop the
heap into a patchwork of free and used blocks, and once no single free
block clears a request's size, that request fails — even though summing
every free block's size would clear it many times over.

This task simulates a heap made of fixed-size blocks (no splitting, no
merging — the simplest setting where fragmentation still bites) and asks:
for each allocation request, does it succeed, and specifically, *why*
would a naive "is there enough free memory overall?" check give the
wrong answer?

## Task

Implement

```cpp
void classify_allocations(const int* block_sizes, int num_blocks,
                           const int* op_types, const int* op_sizes, const int* op_ids,
                           int num_ops, int* out_labels);
```

The heap is `num_blocks` blocks with byte sizes `block_sizes[0..num_blocks)`,
all initially FREE. Run the `num_ops` operations in order, `op_types[i]`
one of:

- **`0` = ALLOC**: request `op_sizes[i]` bytes. Using **first-fit** — scan
  blocks in index order — find the first FREE block with
  `block_sizes[b] >= request`. If one exists, mark it USED and set
  `out_labels[i] = 1`. Otherwise the request fails: `out_labels[i] = 0`,
  and heap state is unchanged (this is the fragmentation case when the
  FREE blocks' sizes summed together *would* have been enough).
- **`1` = FREE**: `op_ids[i]` is the op-index of the ALLOC being released
  (always a prior op that succeeded). Mark the block that ALLOC used FREE
  again.

`out_labels` has length `num_ops` and arrives zero-filled.

## Example

Blocks `{64, 128, 32, 256, 16, 64}`, all used, then three are freed:
block 0 (64), block 2 (32), block 4 (16) — free bytes sum to `112`, but
the *largest* free block is `64`. A request for `100` bytes then fails
(`out_labels[i] = 0`): no single block reaches `100`, even though `112`
total bytes are free. A request for `50` right after succeeds, taking
block 0 first-fit.

## What the gate checks

The driver runs a fixed 17-op scenario — 6 initial allocations fill a
6-block heap, 3 frees fragment it into free blocks of size 64/32/16, then
a mix of requests exercises both an ordinary fail (heap genuinely full)
and two fragmentation fails (request exceeds the largest free block while
the free bytes sum to more) — and prints the label of every ALLOC op in
order. The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{the printed label sequence matches the reference}
$$

The reference's label sequence is `1,1,1,1,1,1,0,1,0,1,1,1,0` (13 ALLOC
ops). A classifier that only checks "is there enough free memory in
total" gets the two fragmentation cases (requests `100` and `40`) wrong —
it would label them `1` instead of `0` — and fails the gate.
