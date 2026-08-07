A researcher is trying to profile the loading time of GGUF models, but they can't accurately distinguish between disk I/O for weights and the parsing overhead of the container itself. We need a zero-dependency parser that reads a GGUF header, extracts all metadata (including complex nested arrays), lists the tensors, and computes exactly how many bytes are wasted in alignment padding.

Your task is to implement `parse_header` and `compute_overhead` in `gguf_parser/parser.py`:

1. **Parse the manifest**: Process the magic bytes, version, and loop through the metadata KVs and tensor information. You must decode all GGUF types (0 through 12, covering all integer sizes, floats, strings, and booleans), including recursively decoding nested arrays. Return a dictionary containing the manifest details.
2. **Compute overhead**: GGUF aligns the start of its tensor data to a block boundary (defaulting to 32 bytes, or explicitly provided by `general.alignment` in the metadata). Compute the container's metadata footprint, the tensor info footprint, and exactly how many bytes of zero-padding are wasted before the tensor data starts.
3. **Safety net**: Write a regression test verifying that `compute_overhead` doesn't over-pad when the header is already perfectly aligned to the boundary.
