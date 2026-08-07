# Symptom: Tool Calls and Constrained Decodes Diverge Under Strict JSON Schemas

Our upstream pipeline uses structured output generation to invoke external tools, but downstream tools frequently throw runtime validation errors or process garbage data.

We observed three distinct failure modes in production:

1. A model configured with `format: "json"` returned syntactically valid JSON, but required fields like `user_id` were missing and numeric types like `port` arrived as strings (`"8080"` instead of `8080`). The pipeline assumed `format: "json"` guaranteed schema compliance, leading to unhandled type errors when executing tool functions.
2. Even when tool calls included non-empty argument strings, parsing them against declared JSON schemas revealed parameter type mismatches, unrecognized extra properties, and missing required properties.
3. High-throughput schema transformations lost precision: converting internal schema definitions into constrained decoding formats and back corrupted pattern constraints and array item definitions, causing schema drift across retries.

We need a unified validation and schema translation layer that handles tool call argument parsing, verifies schema conformance, identifies non-conforming responses even when `format: "json"` is active, and round-trips tool JSON schemas through constrained generation formats without losing structural integrity.
