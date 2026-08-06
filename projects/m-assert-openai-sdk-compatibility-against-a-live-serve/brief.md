# Live Server Compatibility: OpenAI Python SDK vs custom OpenAI-compatible vLLM Server

Our serving integration team has noticed subtle API incompatibilities when clients use the official `openai` Python SDK against our custom vLLM serving endpoint. While basic text completion calls work, edge-case request payload shapes either crash the server or yield unparseable responses on the client side.

Specifically, downstream clients report failures across 8 distinct request shapes:
1. Standard streaming chat completion with delta chunks and `usage` statistics.
2. Structured output requests using JSON Schema response formats (`response_format={"type": "json_object"}`).
3. Tool / function calling requests containing `tools` and `tool_choice`.
4. Multimodal input payloads with mixed inline text and image content parts.
5. Logprobs requests returning top-k log probabilities per token.
6. Multi-choice completion requests with `n > 1`.
7. Prompt formatting with explicit token arrays (`List[int]`) instead of raw string prompts.
8. Chat completions with `stop` sequences supplied as arrays vs single strings.

To eliminate these regressions before our next production release, we need an automated compatibility layer and assertion suite that verifies request transformation, server-side payload handling, and client response deserialization.

Your task is to implement the API transformation mapping in `compat/adapter.py`, construct the compatibility suite in `compat/suite.py`, and author a regression test suite in `tests/test_regression.py` that catches dropped request fields or malformed response shapes.
