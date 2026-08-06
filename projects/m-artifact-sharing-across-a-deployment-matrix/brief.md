# Artifact Sharing Across a Deployment Matrix

## Symptom
Our TensorRT-LLM and ONNX serving infrastructure spent 45 minutes on cold starts during a recent autoscale event because worker nodes repeatedly re-compiled model engines for identical hardware-software tuples. Once live, the fleet experienced elevated p99 latency spikes during traffic bursts. Investigation revealed that nodes were running mismatched artifact variants across GPUs and target microarchitectures, rendering deployment decisions inconsistent. Furthermore, a recent release promoted an engine build that broke tensor layout assumptions on target nodes without being flagged prior to production traffic.

## Goal
You need to build a deployment matrix artifact manager and serving analyzer. First, implement a deterministic artifact resolution system that maps target deployment specs (GPU architecture, precision, CUDA version, tensor dimensions) to reusable compiled engine artifacts, avoiding redundant engine builds across nodes. Second, implement an instance-count analyzer that determines the minimum pool size required to satisfy a target p99 latency SLO under peak traffic distributions. Finally, construct a regression canary gate that validates newly compiled artifacts against reference execution traces before they enter the deployment matrix.

## Structure
* Milestone 1: Implement the artifact repository and matrix resolution logic to select or reuse cached compiled engines matching strict deployment target hashes.
* Milestone 2: Calculate the minimal node instance count needed to meet a target p99 latency SLO given arrival rate parameters and worker processing distributions.
* Milestone 3: Write a canary regression safety test in `tests/test_regression.py` that verifies output tensors and rejects artifacts that violate output layout invariants.
