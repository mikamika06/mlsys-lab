import numpy as np


def estimate_activation_memory_and_comm(seq_len, batch_size, hidden_dim, tp_size, bytes_per_elem=2):
    """Estimate per-rank activation memory and communication volume for TP-only vs TP+SP."""
    S = seq_len
    B = batch_size
    H = hidden_dim
    N = tp_size
    b = bytes_per_elem

    ln_norm_bytes_tp = S * B * H * b
    ln_norm_bytes_sp = (S // N) * B * H * b

    col_input_bytes_tp = S * B * H * b
    col_input_bytes_sp = S * B * H * b

    col_output_bytes = S * B * (H // N) * b

    row_input_bytes = S * B * (H // N) * b
    row_output_bytes_tp = S * B * H * b
    row_output_bytes_sp = (S // N) * B * H * b

    tp_act_bytes = (
        ln_norm_bytes_tp
        + col_input_bytes_tp
        + col_output_bytes
        + row_input_bytes
        + row_output_bytes_tp
    )

    sp_act_bytes = (
        ln_norm_bytes_sp
        + col_input_bytes_sp
        + col_output_bytes
        + row_input_bytes
        + row_output_bytes_sp
    )

    tensor_elements = S * B * H
    tensor_bytes = tensor_elements * b

    comm_factor = (N - 1) / N
    all_reduce_comm_bytes = 2 * comm_factor * tensor_bytes

    all_gather_comm_bytes = comm_factor * tensor_bytes
    reduce_scatter_comm_bytes = comm_factor * tensor_bytes
    sp_total_comm_bytes = all_gather_comm_bytes + reduce_scatter_comm_bytes

    return {
        "tp_activation_bytes": float(tp_act_bytes),
        "sp_activation_bytes": float(sp_act_bytes),
        "tp_comm_bytes": float(all_reduce_comm_bytes),
        "sp_comm_bytes": float(sp_total_comm_bytes),
        "memory_saving_ratio": float(tp_act_bytes / sp_act_bytes),
        "comm_ratio": float(sp_total_comm_bytes / all_reduce_comm_bytes),
    }
