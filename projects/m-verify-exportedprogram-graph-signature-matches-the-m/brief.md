# Export Verification and Serialization Safety in torch.export

## Context & Symptom
Our deployment pipeline converts PyTorch modules into `ExportedProgram` artifacts using `torch.export.export`. During a recent production rollout, a container-mutating module appeared to export successfully, but silent graph-signature drift and missing tensor mutated state caused state dictionary mismatches in downstreams. Furthermore, round-tripped serialized programs (`torch.export.save`/`torch.export.load`) failed numerical verification checks against original PyTorch source modules due to unverified inputs and state signature changes.

To prevent bad artifacts from reaching serving nodes, we need a unified export verification suite that validates graph signatures, verifies round-trip binary save/load numerical fidelity, and distinguishes strict versus non-strict tracing semantics on state-mutating containers.

## Task Overview
Implement the core verification utilities in `export_verify/verifier.py`:

1. **Graph Signature Verification (`verify_graph_signature`)**:
   Verify that an `ExportedProgram`'s `graph_signature` accurately reflects the source module's parameters, buffers, and expected input spec (user inputs vs graph inputs). Return a boolean status along with a detailed dictionary of signature matches.

2. **Round-Trip Serialization & Equivalence (`verify_roundtrip_equivalence`)**:
   Save an `ExportedProgram` to disk via `torch.export.save`, reload it using `torch.export.load`, and execute both the original module and the loaded program on sample inputs. Compare output tensors with `torch.allclose` to guarantee numerical equivalence within tolerance.

3. **Strict vs Non-Strict Export Inspector (`inspect_strict_export_behavior`)**:
   Export a container-mutating module (a module whose forward pass mutates internal container state or input attributes) under both `strict=True` and `strict=False`. Report signature structural differences, mutation safety flags, and execution behavior.

4. **Regression Safeguard (`tests/test_regression.py`)**:
   Write a regression test suite that validates these export invariants. The test suite must actively catch cases where graph signature validation fails to check for parameter/buffer name and shape matches.
