import numpy as np


def compute_pipeline_comm_volume(layer_assignments, tensor_shapes, dtype_bytes=2):
    """Compute per-rank pipeline communication volume in bytes."""
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
