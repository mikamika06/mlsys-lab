# Symptom Report: OpenAI-Compatible API Stream Reassembly, Template Rendering, and Token Accounting Discrepancies

During high-throughput serving validation runs against our vLLM-backed cluster, our offline evaluation pipeline and API proxy layers have started exhibiting severe drift and production anomalies. Engineers report three distinct symptoms affecting request handling and telemetry.

First, when consuming recorded Server-Sent Event (SSE) streams from chat completion endpoints, downstream analytical tools fail to reconstruct complete chat messages. Specifically, responses containing tool calls and multi-chunk streaming deltas result in corrupted JSON payloads, missing tool arguments, or dropped function names, causing downstream execution loops to crash when parsing tool invocations.

Second, our offline request simulation harness is unable to accurately mirror the server-side prompt construction. When pre-computing token budgets or validating prompt injections, the locally rendered chat template strings diverge from the exact byte-for-byte strings processed by the inference engine, leading to mismatched KV-cache pre-allocations and invalid cache hit rates.

Third, our batch token accounting service reports systematic errors in predicted prompt and completion token counts. The billing and rate-limiting subsystems receive inaccurate token volume estimates for batches containing complex multi-turn conversations and tool outputs, resulting in premature context window truncations and erroneous quota enforcement.

These symptoms prevent reliable offline replay, auditing, and cost attribution for our serving infrastructure.
