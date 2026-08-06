import numpy as np


def compute_pipeline_comm_volume(layer_assignments, tensor_shapes, dtype_bytes=2):
    num_ranks = max(layer_assignments) + 1
    send_bytes = np.zeros(num_ranks, dtype=np.int64)
    recv_bytes = np.zeros(num_ranks, dtype=np.int64)

    num_layers = len(layer_assignments)
    for i in range(num_layers - 1):
        r_curr = layer_assignments[i]
        r_next = layer_assignments[i + 1]
        if r_curr != r_next:
            shape = tensor_shapes[i]
            vol = int(np.prod(shape)) * dtype_bytes
            send_bytes[r_curr] += vol
            recv_bytes[r_next] += vol

    return {
        "send_bytes": send_bytes.tolist(),
        "recv_bytes": recv_bytes.tolist(),
        "total_volume": int(np.sum(send_bytes)),
    }


def derive_load_balanced_sharding(num_layers, layer_weights, num_ranks=4):
    weights = np.array(layer_weights, dtype=np.float64)
    n = len(weights)

    dp = np.full((n + 1, num_ranks + 1), fill_value=np.inf)
    parent = np.zeros((n + 1, num_ranks + 1), dtype=int)
    dp[0, 0] = 0.0

    prefix = np.zeros(n + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(weights)

    for k in range(1, num_ranks + 1):
        for i in range(k, n + 1):
            for j in range(k - 1, i):
                cost = prefix[i] - prefix[j]
                max_cost = max(dp[j, k - 1], cost)
                if max_cost < dp[i, k]:
                    dp[i, k] = max_cost
                    parent[i, k] = j

    assignments = np.zeros(n, dtype=int)
    curr = n
    for k in range(num_ranks, 0, -1):
        prev = parent[curr, k]
        assignments[prev:curr] = k - 1
        curr = prev

    return assignments.tolist()


def launch_2rank_ring_all_reduce(tensor_a, tensor_b):
    a = np.array(tensor_a, dtype=np.float32)
    b = np.array(tensor_b, dtype=np.float32)
    expected = a + b
    return {"rank0": expected.copy(), "rank1": expected.copy()}
