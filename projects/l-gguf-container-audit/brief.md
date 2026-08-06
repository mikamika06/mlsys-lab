# Checkpoint audit before the server touches the model

A model server loads GGUF checkpoints from a shared volume that several teams
write to. Twice this month a node has died forty seconds into a load: long
enough to have been taken out of the pool, mapped thirteen gigabytes, and
started faulting pages, but not long enough to have served anything. Both times
the checkpoint was structurally wrong and nothing looked at it before `mmap`.

The fix is an audit that runs in milliseconds on the file's first few kilobytes
and refuses the load with a message naming the tensor at fault. Along the way it
has to be able to read the weights, because "the file parses" and "the weights
are the numbers we shipped" are different claims, and only the second one is
worth a rollback.

You are writing `ggufkit`. No GGUF library is installed and none will be — the
audit runs on the serving node, where the dependency budget is `struct` and
nothing else.

## The file

`projects/_fixtures/gguf/slice.gguf` is real. It was cut out of a quantised
llama checkpoint: the key/value section is that model's, byte for byte, and the
three tensors that came along kept their original ggml types and their original
quantised bytes. Only the number of tensors and the length of the longest arrays
changed, and `manifest.json` records exactly what was cut. `llama.cpp` reads its
header without complaint.

`slice_corrupt.gguf` is the same file with one field changed. That is the
failure you are being asked to catch.

## What you build

`ggufkit/container.py` — the container reader. Header, then the key/value
section with every GGUF type including nested arrays, then the tensor index.
The data section begins at the first `general.alignment` boundary at or after
the end of the index, and each tensor's stored offset is relative to that point,
which is where hand-written readers usually go wrong.

`ggufkit/quants.py` — dequantisation from raw bytes. A Q4_K superblock is 144
bytes describing 256 weights: two half-precision scales, twelve bytes of packed
six-bit sub-scales, and 128 bytes of nibbles. Q6_K is 210 bytes for the same 256
weights, with the six bits of each quant split across two arrays. You also write
the binary16 conversion, because there is no numpy on the serving node either.

Your output is compared against values recorded from the reference
implementation. The reference computes in float32 and so should you be able to:
a correct implementation matches it far inside the tolerance, and the common
mistakes — wrong nibble order, forgetting the `-32` centring in Q6_K, reading
the packed scales with the wrong branch for `j >= 4` — miss it by whole orders
of magnitude, not by rounding.

`ggufkit/plan.py` — what the load would cost. Which pages a set of tensors
touches, how many bytes become resident once you round to page boundaries, how
much of that is waste, and how many pages two tensors share. This is the number
that decides whether lazy per-layer loading is worth building.

`tests/test_container.py` — the check that runs in the serving path.
`assert_loadable(blob)` passes silently on a good container and raises
`AssertionError` naming the offending tensor on a bad one.

## Milestones

1. Header and key/value section, with types reported correctly.
2. Tensor index, alignment arithmetic, absolute offsets; accept the clean file
   and reject the damaged one by name.
3. Q4_K: one superblock, then the whole tensor.
4. Q6_K and F32.
5. The load plan: page accounting, waste, sharing.
6. The regression test, wired the way the server would call it.

## What is not being tested

Speed. This is a correctness audit, and a pure-Python dequantiser is not what
you would ship in the hot path — it is what you run once per checkpoint, on a
few kilobytes of header and one tensor you sampled. Getting the bytes right is
the whole job.
