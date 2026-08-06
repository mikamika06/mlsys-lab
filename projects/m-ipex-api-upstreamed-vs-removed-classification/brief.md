# Intel IPEX API Migration & Op-Graph Optimization Audit

We are upgrading our PyTorch CPU inference stack to recent PyTorch releases and cleaning up legacy Intel Extension for PyTorch (IPEX) dependencies. During a recent maintenance window, an engineer attempted to clean up IPEX API usage, but model throughput degraded on CPU inference, and several microservices threw runtime errors during startup.

An initial audit revealed two core issues:

1. **Deprecated API misclassification:** The codebase contains calls to various `ipex.*` APIs. Some APIs have been fully upstreamed into PyTorch (`torch.*` or `torch.cpu.*`) and should be migrated, while old IPEX experimental APIs were completely removed or renamed and require direct replacement with standard PyTorch constructs. Calling removed functions directly causes `AttributeError` at startup.

2. **Memory layout op-graph mismatch:** To optimize CPU inference, models are converted using `ipex.optimize(...)`. When developers attempted to replace `ipex.optimize` with manual `tensor.to(memory_format=torch.channels_last)` calls, memory overhead increased and operator fusion performance dropped. A closer look showed that `ipex.optimize` performs automated graph-level conversions (e.g., fusing layout changes, converting weight layouts statically, and eliminating runtime copy nodes) that manual tensor layout conversions miss.

Your task is to build an audit tool and optimization graph analysis module:
- Classify a dictionary of IPEX API calls into `upstreamed`, `removed`, or `retained` status, providing their proper PyTorch migration targets.
- Analyze two computational graph representations (one produced via `ipex.optimize` and one via manual `channels_last` conversion) to quantify graph differences, such as eliminated runtime copy nodes (`to(channels_last)`), fused convolution layout changes, and memory footprint reduction.
- Write a suite of regression tests in `tests/test_regression.py` that verifies proper API classification and catches unoptimized op-graphs before deployment.
