Our support queue is flooded with bug reports for our local LLM client using GGUF format and llama.cpp.

One user writes: "I unchecked the `mmap` box in settings, and now the app freezes for 70 seconds before the chat appears, and then crashes with an OOM! I have 8GB of RAM, and the model is 7GB."

Another user complains: "With `mmap` on, the app loads instantly, but when I paste a huge document (setting `batch` to 1024 and `ubatch` to 1024), it crashes exactly when generation starts."

A third user notes: "I changed `ubatch` to 128, and it didn't crash, but my context size is still 8192. I thought `batch` caused the crash?"

We need a memory planning module to predict failures before they occur so we can warn the user and adjust settings automatically. 

Your tasks:
1. Implement `compute_buffer_bytes` and `load_time_seconds` in `memplan/predict.py`. Our engine allocates a compute buffer equal to `ubatch * (hidden + ffn) * 4` bytes. `batch` does not affect this buffer. For load time, `mmap` has a fixed 0.05 seconds overhead, whereas `--no-mmap` reads the entire model into RAM at the disk's bandwidth.
2. Implement `attribute_oom` in `memplan/oom.py`. Engine allocation happens strictly in this order: Weights (only if not using `mmap`, as `mmap` pages lazily), then KV cache, then the compute buffer. Return `"weights"`, `"kv"`, `"compute"`, or `"none"`.
3. Add a test in `tests/test_regression.py` that verifies the exact allocation order of Weights vs KV cache.
