We are implementing core low-level ML system components for processing tokenizer conversion and vocabularies specifically tailored for llama.cpp integration in low-level engines.

The pipeline currently suffers from subtle index mismatches and incorrect metadata classification when mapping BPE tokens or sentencepiece configurations into the expected format. Specifically, when converting or loading specific tokenizers, the merges array needs to be rebuilt precisely in llama.cpp's expected order, matching rank indices and sorting patterns without introducing sorting anomalies.

Additionally, vocabulary artifacts contain specific token types (such as normal, unknown, control, user-defined, unused, or byte tokens) that must be correctly classified and verified. During ingestion, a specific token type often gets misclassified under edge conditions, leading to invalid tokenization behavior or mismatches in vocabulary serialization.

Your task is to implement the core processing routines in `llama_cpp_tok/` that correctly rebuilds the merges array, classifies tokenizer vocabulary types from artifacts, and pinpoints and handles the token with the incorrect token_type. You must also write robust regression tests in `tests/test_regression.py` that enforce these invariants and fail if the merging order or token type classification is corrupted.
