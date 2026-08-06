An engineering task has come up regarding the migration path from TVM Relay legacy to Relax IR. As part of our compiler infrastructure modernizations, we need to analyze how graph representations and IR node counts differ between the older Relay frontend/IR and the modern Relax IR, particularly when compiling a small multi-operator model.

Additionally, during our staging and migration tests, we encountered an unexpected discrepancy in constant folding behavior between Relay and Relax. Specifically, certain constant-folded subexpressions that evaluate cleanly in Relay produce differing constant propagation behavior or fold node counts in Relax under default pass configurations, causing downstream code-generation or shape-insemination mismatches.

Your task is to implement a 3-milestone compiler investigation unit that:
1. Builds a 3-op model representation and accurately computes and compares its Relay IR node count versus its Relax IR node count.
2. Reproduces and verifies the Relay-vs-Relax constant-folding discrepancy using programmatic test structures.
3. Implements a robust regression test suite in `tests/test_regression.py` that fails if the constant-folding check or node-count metric evaluation is improperly stubbed or bypassed.

You must populate the files under the package structure without comments in the code, ensuring all constraints, deterministic references, and milestone checkers pass cleanly.
