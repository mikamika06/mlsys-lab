When converting fine-tuned open-weights models for low-level edge execution in an untranslated MLX runtime on Apple Silicon, downstream inference fails during model initialization. Passing HuggingFace `.safetensors` model weights directly to internal loading logic raises `KeyError` missing key exceptions because parameter key structures do not match the expected module layout. In addition, when cross-validating exported model weights against GGUF format files, developers in zero-dependency container environments cannot install external parsing tools like `safetensors` or `gguf`.

Direct byte reads on `.safetensors` files fail with JSON decoding errors because the 8-byte little-endian header size prefix is not stripped before reading metadata, leading to invalid byte slice bounds when slicing raw tensor payloads.

To resolve these errors, build a zero-dependency parsing and format interoperability module in pure Python and NumPy to:
1. Manually parse `.safetensors` file headers from raw byte buffers by unpacking the uint64 Little-Endian header length, deserializing the JSON string, and extracting precise byte slices for each tensor payload.
2. Parse GGUF binary tensor headers and verify bit-identical `float16` payload representations against safetensors files.
3. Remap HuggingFace tensor names to match target MLX module hierarchies and catch untranslated weight key mismatches before runtime loading.
