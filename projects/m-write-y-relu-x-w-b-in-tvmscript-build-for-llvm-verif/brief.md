# TVM Relax IR & TVMScript Import/Build Pipeline

We are experiencing inconsistencies across our compilation pipeline when ingesting PyTorch models into Apache TVM Relax. Recent production deployments showed unexpected runtime numerical divergence between exported PyTorch computational graphs and compiled TVM artifacts executed via LLVM, alongside discrepancies in operator lowerings during graph transformation passes.

To stabilize our compilation toolchain and verify graph translation fidelity, we need a standardized module and validation suite that covers end-to-end frontend import, graph inspection, and executable compilation.

## Symptoms & Pipeline Issues
1. Manual TVMScript implementations of basic dense layers with activation (such as `y = relu(x @ w + b)`) are producing silent memory errors or wrong results when executed via LLVM runtime backends due to incorrect shape annotations or missing memory bindings.
2. Models imported from PyTorch `torch.export` (`ExportedProgram`) using Relax's `from_exported_program` frontend produce IRModules with varying levels of operator abstraction (raw Relax operators versus call-outs to TensorIR kernels). We lack metric logging and automated verification of how many `R.call_tir` bindings exist in the resulting IRModule compared to raw Relax standard ops.
3. Regression tests are currently absent, allowing breaking changes in graph lowering passes to slip through unnoticed.

## Tasks Required
You must implement a set of robust tools in `tvm_pipeline/ops.py`, `tvm_pipeline/importer.py`, and `tvm_pipeline/inspect.py`:
- Construct a valid TVMScript module representing $y = \text{ReLU}(XW + B)$, compile it to LLVM, and expose an execution interface that matches NumPy outputs within machine precision tolerances.
- Build an importer pipeline using `tvm.relax.frontend.torch.from_exported_program` to ingest a PyTorch MLP model and emit a target TVM Relax `IRModule`.
- Implement an IRModule inspector that traverses Relax functions to count and contrast `R.call_tir` invocations against standard Relax high-level operations.
- Provide a full suite of regression tests in `tests/test_regression.py` that validates linear-ReLU operator equivalence and detects improper graph lowering or operator misclassifications.
