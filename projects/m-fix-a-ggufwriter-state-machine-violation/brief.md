# Ticket: Fix GGUFWriter State-Machine Violation, Peak Memory, and Streaming Tensor Iterator

## Symptom

When building custom workflows that serialise large language models directly into the GGUF format using our low-level `gguf-py` library helpers, downstream consumers and custom writer scripts are encountering sporadic runtime errors and extreme memory overhead.

First, callers are hitting hard state-machine violations inside `GGUFWriter`. Specifically, the writer throws errors when tensors are added or metadata is written out of the expected strict sequence, making it impossible to interleave or dynamically stream chunks of tensors without fully buffering everything upfront.

Second, the current implementation loads entire GGUF files or dumps arrays into memory via full-read operations rather than leveraging memory-mapping techniques. When processing models in the tens of gigabytes, this causes peak memory usage to skyrocket unnecessarily, leading to out-of-memory kills on constrained hardware nodes.

Third, there is currently no robust mechanism to iterate over tensors in a streaming fashion directly from a GGUF file buffer or file descriptor. Instead, users are forced to load all tensor info and raw bytes into contiguous memory at once.

We need to refactor the GGUF writer helper package to relax or correctly manage the internal state-machine constraints, introduce proper memmap versus full-read strategies to control peak memory consumption, and implement a clean, low-overhead streaming GGUF tensor iterator.
