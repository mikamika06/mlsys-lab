# KV Cache Byte Budget and Core ML State Name Repair

During export of a lightweight decoder model for on-device deployment with an in-model Key-Value (KV) cache, runtime allocation failures occur on memory-constrained target devices. Furthermore, when loaded into the Core ML execution engine, stateful tensors fail to bind, reporting `StateType name mismatch` errors or exceeding the allocated hardware memory budget.

Your task is to analyze the model memory requirements, determine the maximum allowable context length given a rigid byte budget, and build conversion/repair utilities that safely export the stateful model to Core ML with aligned state names.

## Key Symptoms
* **Out of Memory (OOM) at Runtime**: Models deployed on target hardware crash during long-context generation because the KV cache memory scale was improperly calculated.
* **Core ML State Binding Failures**: Initializing stateful Core ML execution context raises runtime errors stating that the requested state tensor names do not match the expected `StateType` metadata in the converted ML Model package.
* **Unvalidated Export Integrity**: Upstream changes to model export scripts periodically regress state binding names or recalculate max context lengths incorrectly without failing basic forward pass tests.

## Objectives
1. Implement KV cache byte budget calculation logic to determine the exact maximum context sequence length supported within target memory limits, taking into account precision, head counts, and layer dimensions.
2. Build an export routine that constructs a PyTorch stateful decoder module with in-model KV caching and converts it into a Core ML model package.
3. Implement a state descriptor name repair function that patches mismatched `StateType` metadata within converted models and write a regression test suite that verifies state name alignment and context budget constraints under fault injection.
