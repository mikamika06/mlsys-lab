# Disk-size table for a model across six quantization schemes

Someone computed checkpoint sizes by hand in Excel and picked W4A4 for the
old inference nodes based on those numbers: it had to fit on disk and be the
cheapest of all six supported schemes. It fit. But serving on those nodes
got slower than when they were just running fp16, and nobody can explain
why by looking only at the byte count. That same Excel sheet claims W4A16
and W4A8 weigh different amounts, even though on disk they're the exact
same file at the exact same size.

We need to compute disk size for the six quantization schemes from the
model config instead of by hand, and alongside that check whether the
hardware even has a native path for the chosen bit-width — because that's
where the gap between "fewer bytes" and "faster" hides: without a native
kernel, "4-bit" means unpacking back to fp16 before every matmul.

## What you write

`diskplan/schemes.py`:

```python
tensor_bytes(tensor, scheme) -> int
disk_size(model, scheme) -> int
size_table(model, schemes) -> list[row]
```

`tensor` is `{"name", "kind", "count"}`, where `kind` is `"linear"`,
`"embed"`, or `"norm"`, and `count` is the element count. `scheme` is
`{"name", "bits", "group_size"}`, where `group_size` is the size of the
group sharing one scale factor, or `None`/`0` if there's a single scale
for the whole tensor.

`"embed"` and `"norm"` tensors are always stored in fp16 (`count * 2`
bytes), regardless of scheme — embedding and normalization weights don't
get quantized. If `scheme["bits"] >= 16`, it's also just `count * 2` for
any tensor. Otherwise: the weights themselves take `count * bits` bits,
rounded up to a byte; the number of groups is `1` if `group_size` is
empty, otherwise `count` rounded up to `group_size`; each group adds 2
bytes for its fp16 scale. Total: `weight_bytes + groups * 2`.

`model` is `{"tensors": [tensor, ...]}`. `disk_size` is the sum of
`tensor_bytes` over all tensors in the model. `size_table(model, schemes)`
returns a list in the SAME order as the input `schemes` (the first scheme
in the list is the baseline, usually fp16), unsorted:
`{"scheme", "bits", "bytes", "ratio"}`, where `ratio` is this scheme's
`bytes` divided by the baseline scheme's `bytes`.

`diskplan/hardware.py`:

```python
hardware_native(scheme, hardware) -> bool
gate_table(model, hardware, schemes) -> list[row]
best_native_scheme(model, hardware, schemes) -> str | None
```

`hardware` is `{"native_bits": [int, ...]}` — the bit-widths the hardware
handles with a native kernel, without unpacking to fp16 before the matmul.
`hardware_native` checks whether `scheme["bits"]` is among them.
`gate_table` is the same as `size_table`, plus a `"native"` field on each
row, in the same order. `best_native_scheme` returns the `"name"` of
whichever native scheme gives the smallest `disk_size`; ties go to the
one earlier in the `schemes` list; if no scheme is native, `None`.

## How it's graded

The grader computes the reference itself, from the same config, across
several models, scheme sets, and hardware profiles. The third milestone is
yours: you write a test, and we swap in a version of `tensor_bytes` that
quantizes absolutely everything, `embed` and `norm` included. Your test
needs to catch that.

```
mlsys project start m-disk-size-table-across-six-schemes
mlsys project grade m-disk-size-table-across-six-schemes --milestone 1
```
