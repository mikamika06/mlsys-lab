# Ticket: Per-Layer Sensitivity Ranking and Mixed-Precision Bit Allocation

## Symptom

When quantizing large language models to lower bitwidths (such as 3-bit or 4-bit representations), applying a uniform bit allocation across all transformer layers leads to severe perplexity degradation and accuracy drops on downstream evaluation tasks. Certain attention and feed-forward layers are significantly more sensitive to quantization noise than others, meaning that a uniform quantization strategy wastes precious bit-budget on robust layers while over-compressing sensitive ones.

## Context

Low-level machine learning systems require fine-grained control over model representation formats. To preserve model quality under strict memory footprints, we must evaluate layer sensitivity, allocate bits dynamically according to a global size budget, and emit structured configuration groups that downstream quantization runtimes can execute efficiently. Currently, the pipeline lacks the modules required to compute these sensitivity rankings and translate them into valid mixed-precision allocation plans.

## Objective

Implement the core sensitivity analysis, constrained bit allocation, and grouping logic so that models can be quantized heterogeneously without exceeding memory budgets or violating structural constraints.
