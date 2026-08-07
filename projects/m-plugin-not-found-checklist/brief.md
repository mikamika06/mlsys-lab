# Ticket: Engine Deserialization and Op Resolution Failure in TensorRT Runtime

## Symptom
Production TensorRT runtime pipelines are failing to load compiled engines with cryptic `Plugin read failed` or `INVALID_ARGUMENT: getPluginRegistry` errors during deserialization. Downstream deployment teams report that even when an engine successfully deserializes, execution fails silently or raises dimension mismatch exceptions during inference.

Engine builds for modern custom ops using the TensorRT `IPluginV3` API are intermittently throwing registration errors across environment boundaries, while legacy rewrite scripts fail to properly determine whether a custom layer requires a full custom plugin, a graph rewrite into existing native primitives, or a fallback to standard operations.

## Context & Requirements
To establish robust TensorRT deployment pathways, our custom op pipeline needs an automated diagnosis, serialization, and architecture triage system:

1. **Plugin-Not-Found Checklist**: Implement an automated diagnostic workflow that checks engine metadata against registered plugins, verifies namespace matching, checks field attribute compliance, and flags version mismatches.
2. **Plugin Field Round-Trip**: Implement a strict `IPluginV3` plugin field serialization and deserialization mechanism (`PluginField` / `PluginFieldCollection`) that guarantees exact binary and semantic round-trip state reconstruction across dynamic shapes and layer configurations.
3. **Plugin vs. Rewrite vs. Fallback Decision Matrix**: Implement a decision engine that analyzes target ONNX/TRT graph nodes, performance constraints, and hardware capabilities to automatically classify whether an operation should be compiled as a **Custom TRT Plugin**, **Graph Rewrite** (decomposed into TRT native nodes), or **Standard Fallback**.

A regression test suite must be provided to catch subtle configuration errors, bad serialization buffers, and invalid execution strategy decisions.
