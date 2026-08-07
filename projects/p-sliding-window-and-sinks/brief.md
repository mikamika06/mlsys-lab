# Ticket: Long dialogs exhaust memory and degrade quality

## Symptom
During multi-turn conversation serving, long-context requests quickly deplete the KV cache memory pool, leading to heavy preemption or request rejections. Attempts to save memory by naively dropping old context tokens or applying simple sliding window attention lead to a severe performance drop: perplexity explodes, and the model completely loses track of critical early system instructions and long-range dependencies.

## Goal
Implement a streaming attention mechanism using a sliding window combined with persistent sink tokens (attention anchors). Show that for short contexts within the window size, the sliding-window-with-sinks implementation yields exact logit and attention output equivalence with full standard attention. Demonstrate that on long dialogs, KV cache memory footprint remains bounded under a fixed budget while maintaining retrieval accuracy and stability on context search tasks.

## Deliverables
- Implement `attn/window_sink.py` with support for configurable window sizes and persistent sink tokens.
- Maintain a bounded KV cache manager that retains initial anchor tokens alongside a rolling window of recent context.
- Verify exact output matching on short sequences and bound memory growth on arbitrary-length streams.
- Ensure proper behavior at window boundaries when context transitions from prefill to step-by-step generation.
- Supply a comprehensive regression suite in `tests/test_regression.py` that catches common implementation pitfalls such as dropping sink tokens or incorrect KV cache index shifting.
