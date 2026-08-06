import numpy as np
import ref


def check(workdir):
    import sys

    sys.path.insert(0, workdir)
    from kvfp8.bytes import compute_kv_cache_bytes, max_context_length

    errs = []
    for cfg in ref.CONFIGS:
        b_ref = ref.ref_compute_kv_cache_bytes(
            cfg["num_layers"],
            cfg["num_kv_heads"],
            cfg["head_dim"],
            cfg["seq_len"],
            cfg["batch_size"],
            fp8=False,
        )
        b_got = compute_kv_cache_bytes(
            cfg["num_layers"],
            cfg["num_kv_heads"],
            cfg["head_dim"],
            cfg["seq_len"],
            cfg["batch_size"],
            fp8=False,
        )
        errs.append(abs(b_ref - b_got) / (abs(b_ref) + 1e-12))

        b_ref_8 = ref.ref_compute_kv_cache_bytes(
            cfg["num_layers"],
            cfg["num_kv_heads"],
            cfg["head_dim"],
            cfg["seq_len"],
            cfg["batch_size"],
            fp8=True,
        )
        b_got_8 = compute_kv_cache_bytes(
            cfg["num_layers"],
            cfg["num_kv_heads"],
            cfg["head_dim"],
            cfg["seq_len"],
            cfg["batch_size"],
            fp8=True,
        )
        errs.append(abs(b_ref_8 - b_got_8) / (abs(b_ref_8) + 1e-12))

        m_ref = ref.ref_max_context_length(
            cfg["gpu_mem"],
            cfg["weights"],
            cfg["act_budget"],
            cfg["num_layers"],
            cfg["num_kv_heads"],
            cfg["head_dim"],
            cfg["batch_size"],
            fp8=False,
        )
        m_got = max_context_length(
            cfg["gpu_mem"],
            cfg["weights"],
            cfg["act_budget"],
            cfg["num_layers"],
            cfg["num_kv_heads"],
            cfg["head_dim"],
            cfg["batch_size"],
            fp8=False,
        )
        errs.append(abs(m_ref - m_got) / (abs(m_ref) + 1e-12))

        m_ref_8 = ref.ref_max_context_length(
            cfg["gpu_mem"],
            cfg["weights"],
            cfg["act_budget"],
            cfg["num_layers"],
            cfg["num_kv_heads"],
            cfg["head_dim"],
            cfg["batch_size"],
            fp8=True,
        )
        m_got_8 = max_context_length(
            cfg["gpu_mem"],
            cfg["weights"],
            cfg["act_budget"],
            cfg["num_layers"],
            cfg["num_kv_heads"],
            cfg["head_dim"],
            cfg["batch_size"],
            fp8=True,
        )
        errs.append(abs(m_ref_8 - m_got_8) / (abs(m_ref_8) + 1e-12))

    max_rel_err = float(np.max(errs))
    return {"rel_err": max_rel_err}
