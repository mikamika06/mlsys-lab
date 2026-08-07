# Ticket: Disaggregated KV Cache Handshake Failures and Transfer Latency Anomalies

We are seeing sporadic production failures in our disaggregated prefill-to-decode pipeline (`disagg-kv-v2`). Instances configured as producer-consumer pairs occasionally hang indefinitely during the initial handshake phase or experience severe throughput drops during layer-wise KV cache transfer over RDMA/TCP connections.

## Reported Symptoms

1. **Stuck Handshake:** When starting a prefill producer instance alongside a decode consumer instance, both engines occasionally report timeout errors or hang waiting for peer acknowledge packets. The logs show conflicting parameters (e.g., mismatching layer counts, inconsistent block sizes, or inverted buffer roles), but operators cannot easily tell which side misconfigured its setup.
2. **Log Triage Ambiguity:** On failed transfers, logs from producer and consumer nodes are interleaved in centralized telemetry. On-call engineers currently lack a quick diagnostic tool to parse paired logs and identify exact root causes (such as port collisions, mismatched memory pools, or out-of-order sequence ACKs).
3. **Pipelining Inefficiencies:** In running instances, KV cache transfers between layers do not achieve expected speedups. The scheduling heuristic currently treats transfer overhead as flat rather than modeling the overlap between prefill compute execution, networking bandwidth, and decode memory absorption.

## Deliverables

You need to implement a KV transfer validation and performance modeling module under `kvtransfer/`:

1. `kvtransfer/config.py`: Implement `validate_pair(producer_cfg, consumer_cfg)` to validate that a producer and consumer configuration pair are structurally compatible for disaggregated KV streaming, returning structured errors when invariants are violated.
2. `kvtransfer/triage.py`: Implement `diagnose_stuck_handshake(producer_logs, consumer_logs)` to analyze log events from both sides of a paired run, pinpointing the precise failure mode of a stuck handshake.
3. `kvtransfer/model.py`: Implement `estimate_pipelined_transfer_time(model_cfg, net_cfg)` to calculate the layer-wise pipelined execution and transfer latency, returning total time and speedup ratios over unpipelined transfers.
4. `tests/test_regression.py`: Provide regression tests that catch bad configurations and verify that your triage and performance modeling logic detect invalid state transitions.
