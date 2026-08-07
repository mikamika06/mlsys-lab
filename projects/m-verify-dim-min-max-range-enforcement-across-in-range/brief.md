Ticket ID: COMP-9142
Priority: High
Title: Trace constraint validation fails with obscure errors and allows shape drift

The compilation team reported a critical bug where exported model modules are failing to properly enforce and validate batch size constraints during tracing. Currently, if an end-user traces a module with out-of-range batch sizes, the system panics with an unhelpful `ConstraintViolationError`. It just dumps a raw stack trace without telling the user how to fix their shape environment bounds. We need this exception to natively support a `suggested_fix()` message indicating exactly which dimension failed, what size it received, and what the expected `[min, max]` interval was, so developers can fix their inputs.

Secondly, the compiler's logic for resolving ambiguous shape signatures across multiple forward passes is flawed. `Dim.AUTO`, `Dim.DYNAMIC`, and explicit bounded `Dim` instances are not strictly differentiated. When a dimension is tagged as `Dim.AUTO`, the compiler should aggressively enforce that its shape remains completely static (identical) across all observed batches, failing compilation if it sees any variance. Instead, it seems to be silently adapting. `Dim.DYNAMIC`, however, must compute the tightest `[min, max]` bounding box based on the observed sizes. Explicit dimensions must just verify that all observed sizes fit their bounds.

We need a patch that correctly enforces these range invariants and accurately resolves the signature bounds across in-range and out-of-range batch observations.
