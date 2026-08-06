# Incident Report: ONNX Runtime Startup Latency and Graph Optimization Overhead

## Symptom
Production microservices serving ONNX models have reported severe startup time degradation and sporadic high-memory spikes during container initialization. Profiling reveals that ONNX Runtime online graph optimization at level 3 (`ORT_ENABLE_ALL`) incurs high CPU overhead during session creation, while lower optimization levels finish almost instantaneously but produce suboptimal execution graphs. Furthermore, several deployed models suffer from unnecessary offline model conversion steps when the performance gain from maximum fusion is negligible compared to lower optimization levels.

## Task
We need a diagnostic and selection tool in `ortopt` to analyze ORT graph optimization tradeoffs:
1. In `ortopt/levels.py`, implement `select_cheapest_level` to pick the lowest optimization level whose estimated execution latency is within 5% of the absolute best level's latency.
2. In `ortopt/fused.py`, implement `count_fused_nodes` to parse graph node definitions and count nodes belonging to the `com.microsoft` fused operator domain across different optimization outputs.
3. In `ortopt/costs.py`, implement `evaluate_offline_vs_online` to calculate total operational cost given online optimization overhead vs pre-compiled offline graph loading costs across execution requests.
4. In `tests/test_regression.py`, write regression tests that verify level selection logic and detect regressions when fusion domain counting or threshold selection algorithms are altered.
