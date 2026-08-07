# Ticket: Local Runner Integration Fails Stream Assembly and Overhead Analysis

## Symptom

Our local inference runner integration pipeline is experiencing intermittent failures and performance discrepancies when processing raw endpoint transcripts and constructing prompt templates. Specifically, streaming responses captured from local engine endpoints via line-delimited JSON chunks are failing to reconstruct into cohesive text strings under high concurrency, resulting in truncated outputs or corrupted sentence boundaries in downstream consumers.

Additionally, telemetry reports show that payload estimation tools are incorrectly calculating the token and byte overhead between raw generation requests and structured chat endpoints, causing unexpected context window overflow errors and misconfigured budget allocations during prompt formulation. Furthermore, developers implementing specialized code completion features are encountering malformed fill-in-the-middle payloads where suffix handling and insertion boundaries are incorrectly positioned, leading to syntax errors and invalid token sequences when evaluated by the underlying model runtime.

We need a robust, deterministic module that handles NDJSON stream reassembly correctly, computes precise metrics for request type overheads, and formats fill-in-the-middle requests reliably without introducing regressions or boundary corruption.
