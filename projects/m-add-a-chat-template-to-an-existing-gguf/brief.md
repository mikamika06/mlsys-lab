We provide an internal utility script for researchers to rapidly test new instruction formats by injecting custom chat templates into existing GGUF models. Because model files can be dozens of gigabytes, modifying the metadata in place or doing a quick read-write loop is much faster than running the full quantization pipeline again.

However, a serious symptom has been reported in the issue tracker. Researchers state that when they experiment with a custom prompt format and later decide to revert the model to its original state, they cannot. Their original, canonical template is completely gone. 

Worse, if a user runs our script twice—say, first trying a "ChatML" template, and then a "Llama-3" template—the model permanently loses not just the original template, but also the first experimental one. We need a robust mechanism that automatically preserves the oldest original template when doing these in-place metadata modifications using the `gguf` Python library's structure.

You must implement extraction and safe modification routines that interface with `GGUFReader` and `GGUFWriter` objects, ensuring the first template is always backed up and never overwritten by subsequent modifications. Finally, you will write a regression test to prevent regressions on this backup behavior.
