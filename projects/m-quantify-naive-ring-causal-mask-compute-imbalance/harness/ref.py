import numpy as np


def get_test_cases():
    return [
        {"seq_len": 512, "num_ranks": 4},
        {"seq_len": 1024, "num_ranks": 8},
        {"seq_len": 256, "num_ranks": 2},
    ]


def compute_imbalance(seq_len, num_ranks):
    block_size = seq_len // num_ranks
    workloads = []
    for r in range(num_ranks):
        active_blocks = 0
        for step in range(num_ranks):
            k_block = (r - step) % num_ranks
            if k_block <= r:
                active_blocks += 1
        workloads.append(active_blocks)
    max_w = float(max(workloads))
    avg_w = float(sum(workloads)) / float(num_ranks)
    imbalance_ratio = max_w / avg_w if avg_w > 0 else 1.0
    return {
        "workloads": workloads,
        "max_workload": max_w,
        "avg_workload": avg_w,
        "imbalance_ratio": imbalance_ratio,
    }


def single_process_reference(q, k, v):
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, k.T) * scale
    seq_len = q.shape[0]
    mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
    scores = scores + mask
    max_val = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_val)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    attn = exp_scores / sum_exp
    return np.matmul(attn, v)


def ring_attention_simulate(q, k, v, num_ranks):
    seq_len = q.shape[0]
    chunk_size = seq_len // num_ranks
    scale = 1.0 / np.sqrt(q.shape[-1])

    output = np.zeros_like(q)
    running_max = np.full((seq_len, 1), -1e9)
    running_sum = np.zeros((seq_len, 1))
    running_acc = np.zeros_like(q)

    k_ring = [k[i * chunk_size:(i + 1) * chunk_size] for i in range(num_ranks)]
    v_ring = [v[i * chunk_size:(i + 1) * chunk_size] for i in range(num_ranks)]

    for r in range(num_ranks):
        q_chunks = [q[i * chunk_size:(i + 1) * chunk_size] for i in range(num_ranks)]
        for step in range(num_ranks):
            rank_idx = (r + step) % num_ranks
            kv_idx = (r - step) % num_ranks
            if kv_idx <= rank_idx:
                qc = q_chunks[rank_idx]
                kc = k_ring[kv_idx]
                vc = v_ring[kv_idx]

                scores = np.matmul(qc, kc.T) * scale
                if rank_idx == kv_idx:
                    local_len = qc.shape[0]
                    mask = np.triu(np.full((local_len, local_len), -1e9), k=1)
                    scores = scores + mask

                block_max = np.max(scores, axis=-1, keepdims=True)
                exp_scores = np.exp(scores - block_max)
                block_sum = np.sum(exp_scores, axis=-1, keepdims=True)

                start_idx = rank_idx * chunk_size
                end_idx = (rank_idx + 1) * chunk_size

                curr_max = running_max[start_idx:end_idx]
                curr_sum = running_sum[start_idx:end_idx]
                curr_acc = running_acc[start_idx:end_idx]

                new_max = np.maximum(curr_max, block_max)
                correction_old = np.exp(curr_max - new_max)
                correction_new = np.exp(block_max - new_max)

                new_sum = curr_sum * correction_old + block_sum * correction_new
                new_acc = curr_acc * correction_old + np.matmul(exp_scores, vc) * correction_new

                running_max[start_idx:end_idx] = new_max
                running_sum[start_idx:end_idx] = new_sum
                running_acc[start_idx:end_idx] = new_acc

    for i in range(num_ranks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        output[start_idx:end_idx] = running_acc[start_idx:end_idx] / running_sum[start_idx:end_idx]

    return output
