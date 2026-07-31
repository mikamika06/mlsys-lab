# A homegrown GGUF converter

We're shipping a model with a nonstandard architecture. No off-the-shelf converter
knows it, and we're not going to patch someone else's repo for every release. We
need our own container writer — one that produces files every engine that
understands the format can read, no exceptions.

We also need a reader alongside it: without one there's no way to check our own
output, let alone figure out why someone else's file won't load.

## Format

A file is four parts back to back: header, metadata, tensor table, data section.

**Header.** `GGUF` (4 bytes), version `uint32`, tensor count `uint64`,
metadata pair count `uint64`. All little-endian.

**Metadata.** Each pair: a string key, then a `uint32` type, then the value.
A string is a `uint64` length followed by that many bytes of UTF-8. Type numbers:
`0 uint8, 1 int8, 2 uint16, 3 int16, 4 uint32, 5 int32, 6 float32, 7 bool,
8 string, 9 array, 10 uint64, 11 int64, 12 float64`. An array is an element
type `uint32`, a count `uint64`, then the elements back to back.

**Tensor table.** Per tensor: a name string, number of dimensions `uint32`,
the dimensions themselves as `uint64`, ggml type `uint32`, offset `uint64`.
Dimensions are stored in the reverse order from what numpy expects.

**The data section** starts after the end of the table is padded up to a
`general.alignment` boundary (default 32). Each tensor's offset is counted
**from the start of this section**, not from the start of the file, and is
also a multiple of the alignment.

## What you write

`gguf_lab/reader.py`:

```python
read_header(path) -> {"magic", "version", "tensor_count", "kv_count"}
read_metadata(path) -> dict
read_tensor_info(path) -> [{"name", "shape", "dtype", "ggml_type", "offset"}, ...]
read_tensor_data(path, name) -> numpy array
```

`gguf_lab/writer.py`:

```python
write(path, arch, metadata: dict, tensors: [{"name", "data"}], alignment=32) -> path
```

A file whose magic isn't `GGUF` must be rejected, not read as garbage.
Requesting a tensor that doesn't exist is also an error, not an empty array.

## How it's graded

The oracle is the **real `gguf` library**, not our own implementation. First
it writes files, and you read them. Then the other way round: you write, it
reads, and every piece of metadata and every tensor must survive unchanged.
On the last milestone we break your own output two ways — truncating the
tail of the file and lying in the header about the tensor count — and your
validator has to catch both.

```
mlsys project start p-gguf-container-from-scratch
mlsys project grade p-gguf-container-from-scratch --milestone 1
```
