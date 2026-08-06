# BLS vs Static Ensemble: Reconstruct When Conditional Routing Is Required

Our serving pipeline for real-time request classification and multi-stage inference has hit a performance wall after transitioning to complex dynamic models.

Under heavy load, client latency spikes unpredictably and inference queues back up. A post-mortem of our production deployment revealed two major issues:

1. Static Triton Ensembles are currently configured for dynamic routing paths where execution branches based on request payloads. However, Triton Ensemble DAGs require static execution graphs, causing failed requests, unnecessary execution of unselected branches, or invalid tensor routing.
2. An initial attempt to migrate to Triton Business Logic Scripting (BLS) resulted in memory leaks and missing execution metrics because model inputs were not properly decoupled from request memory pools across conditional calls.

Your task is to analyze when static Triton ensembles fail, reconstruct the routing pipeline using Triton BLS with proper input/output tensor dynamic allocation, and add a test suite that detects static ensemble routing violations.

You need to implement the BLS routing pipeline in `bls_router/router.py`, measure execution efficiency and routing guarantees in `bls_router/metrics.py`, and construct a regression test in `tests/test_regression.py` that fails if dynamic conditional routing is incorrectly executed via static ensemble DAG structures.
