# Ticket: High inter-service latency in multi-model serving pipeline

Our inference infrastructure processes user queries through three microservices deployed as standalone endpoints: a preprocessor service (performing text cleaning and tokenization), a neural model inference service, and a postprocessor service (applying output formatting and response generation).

Telemetry logs show that over 50% of the total request latency is spent on inter-service network transfers and serialization/deserialization cycles between these microservices. Transferring heavy intermediate arrays and JSON payloads back and forth across RPC boundaries creates severe latency bottlenecks and increases overall network overhead.

We need to collapse these distributed microservice boundaries by moving to a server-side ensemble pipeline using Business Logic Scripting (BLS) within the serving framework. The goal is to consolidate the entire pipeline so that data transfers between preprocessing, model execution, and postprocessing happen in-process via shared memory references, avoiding RPC serialization and cross-network transport entirely.

Your task is to:
1. Define the multi-model architecture as a validated Directed Acyclic Graph (DAG) ensemble pipeline.
2. Implement in-process Business Logic Scripting (BLS) orchestration to execute the pipeline stages without network transport.
3. Quantify data transfer savings comparing inter-service payloads against zero-copy in-process composition.
4. Verify end-to-end output identity between distributed RPC calls and in-process BLS execution.
5. Ensure end-to-end latency remains below the target SLA threshold under identical workloads.
6. Build regression tests for fault tolerance to handle graceful degraded execution when a pipeline stage fails.
