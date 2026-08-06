We're seeing major performance regressions and weird token behavior when porting our inference backend to a hand-rolled loop for a new custom sampler we are experimenting with. Also, some model repos crash immediately on startup when processing user messages.

The C++ `LLMPipeline.generate` path runs at ~50 tokens/second, but our Python prototype of the generation loop is crawling at 2 tokens/second. The prototype loop is doing standard greedy decoding, but it seems to be paying some massive overhead on every single token. We need to measure the baseline throughput delta between the two, because if the overhead is fundamental, we might have to write the sampler in C++ instead of Python.

Worse, the tokens coming out of the manual loop don't match the built-in pipeline when given the exact same prompt and parameters. We need absolute token-id parity for our greedy path before we can even test the custom logic.

Finally, when users deploy models without a `tokenizer_config.json` chat template, the pipeline throws a C++ `RuntimeError`. We need to catch this gracefully and fall back to a standard newline-delimited prompt string rather than taking down the entire service container.
