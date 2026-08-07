# Verify Byte-Fallback Round-Trip Over Rare Unicode

In low-level LLM tokenization pipelines—specifically those using byte-level or byte-fallback BPE (such as llama.cpp and GGUF tokenizer conversions)—rare or unmapped Unicode code points must be safely converted into sequence byte tokens and reconstructed back into exact UTF-8 strings without corruption or loss.

When tokenizing uncommon Unicode sequences (such as rare CJK ideographs, emoji sequences with zero-width joiners, or ancient scripts), the vocabulary may lack matching merged tokens. The tokenizer must fall back to encoding raw bytes using fallback byte tokens (e.g., `<0xXX>`). During decoding, these fallback byte tokens must be gathered, reassembled into raw byte streams, and decoded back into valid UTF-8.

You are tasked with implementing and verifying a robust byte-fallback encoder and decoder module.

## Symptoms & Findings
- Rare or out-of-vocabulary Unicode strings fail to survive the encode-decode round-trip, producing `UnicodeDecodeError` or mutated replacement characters (`\ufffd`).
- Byte fallback tokens are produced correctly during encoding, but decoding improperly treats each byte token as an isolated character rather than accumulating adjacent byte tokens into multi-byte UTF-8 sequences.
- Partial or broken byte sequences across token boundaries cause standard string concatenation decoders to throw exceptions instead of recovering cleanly.

## Deliverables
Implement the tokenizer byte-fallback logic in `bytefallback/convert.py`:
1. `encode_with_fallback(text: str, vocab: dict[str, int]) -> list[int]`: Tokenizes text using standard vocabulary tokens where available, falling back to `<0xXX>` byte tokens for any character or byte sequence missing from the vocabulary.
2. `decode_with_fallback(token_ids: list[int], inv_vocab: dict[int, str]) -> str`: Decodes a sequence of token IDs back into a Unicode string. Merges adjacent byte-fallback tokens (`<0xXX>`) into raw byte buffers before UTF-8 decoding to preserve valid multi-byte character boundaries.
3. `verify_round_trip(text: str, vocab: dict[str, int]) -> bool`: Runs an end-to-end verification that encoding and then decoding any rare Unicode string reproduces the original string perfectly.
4. `tests/test_regression.py`: Unit tests verifying that byte-fallback handles single-byte, multi-byte, rare Unicode, and mixed sequence round-trips without loss, and catches improper byte aggregation bugs.
