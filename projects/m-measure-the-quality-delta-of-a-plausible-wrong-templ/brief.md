Our local model runner service is intermittently failing or producing degraded outputs when handling structured tool calls and custom system prompts, especially under varying context lengths and multi-turn chat configurations. Specifically, during testing with a set of known user queries and tool definitions, we noticed that one of our prompt templates produces a quality delta compared to the reference oracle template, failing to format tool definitions or properly trigger stop sequences. This causes generation loops to hang or emit malformed JSON arguments instead of valid structured calls.

We need to build a diagnostic and validation unit to systematically measure template fidelity. This unit must implement a module that:
1. Computes the quality delta between a plausible wrong prompt template and the golden reference template using structured scoring on reference inputs.
2. Renders complex tool definitions directly into the prompt template and validates that the emitted model generation correctly parses into a structured tool call.
3. Detects prompt templates whose configured stop sequences fail to terminate generation, causing runaway token generation past the stop boundary.

Your task is to implement the components in `templater/` and write a regression test suite in `tests/test_regression.py` that verifies stop sequence and templating invariants. The system must correctly identify faulty templates, validate rendered tool calls against expected schemas, and catch broken template behavior under monkeypatched conditions.
