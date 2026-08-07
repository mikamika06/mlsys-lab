import sys
import os

sys.path.insert(0, os.path.abspath("reference"))

import kvtransfer.config as config
import kvtransfer.triage as triage
import kvtransfer.model as model

validate_pair = config.validate_pair
diagnose_stuck_handshake = triage.diagnose_stuck_handshake
estimate_pipelined_transfer_time = model.estimate_pipelined_transfer_time

CONFIG_PAIRS = [
    (
        {
            "role": "producer",
            "num_layers": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "block_size": 16,
            "dtype": "float16",
            "transport": {"type": "rdma", "gid": "fe80::1", "qp_num": 101}
        },
        {
            "role": "consumer",
            "num_layers": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "block_size": 16,
            "dtype": "float16",
            "transport": {"type": "rdma", "gid": "fe80::2", "qp_num": 102}
        }
    ),
    (
        {
            "role": "producer",
            "num_layers": 64,
            "num_kv_heads": 16,
            "head_dim": 64,
            "block_size": 32,
            "dtype": "bfloat16",
            "transport": {"type": "tcp", "port": 8080}
        },
        {
            "role": "consumer",
            "num_layers": 64,
            "num_kv_heads": 16,
            "head_dim": 64,
            "block_size": 32,
            "dtype": "bfloat16",
            "transport": {"type": "tcp", "port": 8080}
        }
    ),
    (
        {
            "role": "producer",
            "num_layers": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "block_size": 16,
            "dtype": "float16",
            "transport": {"type": "rdma", "gid": "fe80::1", "qp_num": 100}
        },
        {
            "role": "consumer",
            "num_layers": 16,
            "num_kv_heads": 8,
            "head_dim": 128,
            "block_size": 16,
            "dtype": "float16",
            "transport": {"type": "rdma", "gid": "fe80::2", "qp_num": 101}
        }
    ),
    (
        {
            "role": "producer",
            "num_layers": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "block_size": 16,
            "dtype": "float16",
            "transport": {"type": "rdma", "gid": "fe80::1", "qp_num": 100}
        },
        {
            "role": "consumer",
            "num_layers": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "block_size": 16,
            "dtype": "float16",
            "transport": {"type": "tcp", "port": 9000}
        }
    )
]

LOG_TRIAGE_CASES = [
    (
        [{"event": "INIT", "session_id": "s1"}, {"event": "PARAM_SYNC", "mem_key": "k1"}, {"event": "READY"}],
        [{"event": "INIT", "session_id": "s1"}, {"event": "PARAM_SYNC", "mem_key": "k1"}, {"event": "READY"}]
    ),
    (
        [{"event": "INIT", "session_id": "s1"}],
        [{"event": "INIT", "session_id": "s2"}]
    ),
    (
        [{"event": "INIT", "session_id": "s1"}, {"event": "MAGIC_ACK", "status": "REJECTED"}],
        [{"event": "INIT", "session_id": "s1"}]
    ),
    (
        [{"event": "INIT", "session_id": "s1"}, {"event": "PARAM_SYNC", "mem_key": "k1"}],
        [{"event": "INIT", "session_id": "s1"}, {"event": "PARAM_SYNC", "mem_key": "k2"}]
    ),
    (
        [{"event": "INIT", "session_id": "s1"}, {"event": "CONNECT", "status": "TIMEOUT"}],
        [{"event": "INIT", "session_id": "s1"}]
    )
]

MODEL_CASES = [
    (
        {"num_layers": 32, "layer_compute_ms": 1.5, "layer_kv_bytes": 2097152},
        {"bandwidth_gbps": 100.0, "latency_ms": 0.01}
    ),
    (
        {"num_layers": 80, "layer_compute_ms": 4.0, "layer_kv_bytes": 8388608},
        {"bandwidth_gbps": 40.0, "latency_ms": 0.05}
    )
]
