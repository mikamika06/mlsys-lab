# Emit a GGUF with GGUFWriter matching a reference

During local deployment of LLMs with `llama.cpp`, model files must follow the binary GGUF format specification, which encodes metadata key-value pairs followed by tensor headers and aligned tensor data. When exporting modified models or quantizing existing checkpoints, subtle discrepancies in GGUF writer implementations—such as incorrect string padding, miscalculated array offsets, or inconsistent tensor byte alignment—lead to unparseable files or degraded inference outputs. Furthermore, post-processing tools often require updating metadata (such as model hyper-parameters or architecture tags) in place without corrupting or rewriting gigabytes of heavy tensor payload blocks.

You are tasked with building a lightweight GGUF binary tool suite. Your suite must construct valid GGUF binary streams using a structured writer class matching canonical specifications, inspect and dump GGUF files to a standardized structured JSON format equivalent to `gguf_dump --json`, and patch metadata key-value pairs directly in place while strictly preserving raw tensor binary payloads and alignment boundaries.

Implement the GGUF writing, parsing, and patching logic under `ggftool/` and ensure your suite passes all regression test requirements under `tests/test_regression.py`.
