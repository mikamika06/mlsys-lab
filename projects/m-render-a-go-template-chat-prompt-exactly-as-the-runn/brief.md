# Ticket: Local Runner Chat Prompt Mismatches and Template Desyncs

## Symptom Report
We are experiencing intermittent tokenization failures and unexpected model behavior when deploying quantized models across different local runtime environments. Specifically, when local runners execute chat completions using custom Modelfiles, the generated prompt strings frequently diverge from what the underlying model expects, leading to syntax errors or missing special control tokens such as role delimiters.

Additionally, field engineers have reported that model cards downloaded from upstream repositories often specify token configurations that conflict with explicit definitions embedded within local Modelfiles. When operators attempt to verify these discrepancies using template inspection tools, the recovered chat templates from GGUF binary metadata structures occasionally fail to match standard outputs produced by management utilities like ollama show.

This desynchronization causes chat applications to drop user turns, misinterpret system prompts, or hallucinate conversation boundaries due to incorrect template expansion. We need a robust, deterministic suite of low-level utilities to accurately render template strings, audit token mismatches between configuration layers, and parse embedded GGUF chat metadata precisely to ensure runtime consistency across all serving nodes.
