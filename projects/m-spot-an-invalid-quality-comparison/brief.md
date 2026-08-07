# Invalid Quality Comparison and HellaSwag Evaluation

Our evaluation dashboard recently flagged a series of anomalous benchmarking reports where LLM quantization variants were ranked strictly by perplexity (PPL) and raw accuracy without accounting for variance, tokenization mismatches, or context window clipping.

Specifically:
- Model quality comparisons are being reported across disparate tokenizers and non-standardized context lengths, leading to invalid ranking assertions.
- Standard benchmark scripts (such as HellaSwag runs) are yielding point-estimate scores without statistical confidence intervals or standard error bars, masking overlapping confidence bounds.

You need to implement a quality audit module and an evaluation runner that:
1. Validates whether two evaluation setups allow for a statistically sound quality comparison.
2. Computes HellaSwag accuracy alongside standard error bounds using sample variance over task items.
3. Provides regression tests to ensure invalid comparison claims are caught before reports are generated.
