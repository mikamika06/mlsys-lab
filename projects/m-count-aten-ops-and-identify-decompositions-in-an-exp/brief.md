Engineers working on our low-level compiler pipeline are experiencing unexpected compilation failures and verification gaps during model export. While PyTorch's `torch.export` mechanisms power this step, we cannot directly run them in this isolated CI environment. Instead, our instrumentation dumps the exported program graph nodes and tracing events into JSON-like structures.

When attempting to run models containing custom submodules or side effects, the export step either fails with opaque tracebacks or fails to accurately count and classify low-level `aten` operator nodes in the resulting exported program. This lack of visibility makes it difficult to verify whether crucial compiler decomposition passes (such as lowering higher-order operators) successfully executed before backend code generation.

Furthermore, when models attempt to modify external global state, the export system throws errors that are currently swallowed, lacking the precise diagnostic structure required by our telemetry guardrails.

We need a robust analysis module that can programmatically inspect an exported program trace, accurately count `aten` operators, verify that specific decompositions took place (target ops are completely absent), and cleanly capture and categorize export-time errors caused by forbidden global-state mutations.
