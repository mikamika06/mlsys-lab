import time
import numpy as np


def compare_lora_dora_metrics(model_config, batch_data, r=8):
    """Compare memory and step time metrics between LoRA and DoRA fine-tuning runs."""
    in_dim = model_config["in_features"]
    out_dim = model_config["out_features"]
    seq_len = batch_data.get("seq_len", 128)
    batch_size = batch_data.get("batch_size", 4)
    steps = batch_data.get("steps", 10)

    np.random.seed(42)
    x = np.random.randn(batch_size, seq_len, in_dim).astype(np.float32)
    w_base = np.random.randn(out_dim, in_dim).astype(np.float32) * 0.02

    def run_simulation(is_dora):
        lora_a = np.random.randn(r, in_dim).astype(np.float32) * 0.01
        lora_b = np.zeros((out_dim, r), dtype=np.float32)
        scale = 2.0
        m_vec = np.linalg.norm(w_base, axis=1, keepdims=True) if is_dora else None

        start_time = time.perf_counter()
        tot_mem = 0
        for _ in range(steps):
            delta = (lora_b @ lora_a) * scale
            if not is_dora:
                w_eff = w_base + delta
                out = x @ w_eff.T
            else:
                w_comb = w_base + delta
                norm_w = np.linalg.norm(w_comb, axis=1, keepdims=True)
                w_eff = m_vec * (w_comb / norm_w)
                out = x @ w_eff.T

            grad_out = out * 0.01
            grad_w_eff = np.einsum("bsi,bsj->ij", grad_out, x)
            grad_b = (grad_w_eff @ lora_a.T) * scale
            grad_a = (lora_b.T @ grad_w_eff) * scale

            lora_b -= 0.001 * grad_b
            lora_a -= 0.001 * grad_a

            act_mem = x.nbytes + out.nbytes + grad_out.nbytes
            param_mem = lora_a.nbytes + lora_b.nbytes + w_base.nbytes
            if is_dora:
                param_mem += m_vec.nbytes
            tot_mem += act_mem + param_mem

        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / steps
        avg_mem = tot_mem / steps
        return avg_time, avg_mem

    lora_time, lora_mem = run_simulation(is_dora=False)
    dora_time, dora_mem = run_simulation(is_dora=True)

    return {
        "lora": {"avg_step_time": lora_time, "avg_memory_bytes": lora_mem},
        "dora": {"avg_step_time": dora_time, "avg_memory_bytes": dora_mem},
        "time_ratio_dora_vs_lora": dora_time / lora_time if lora_time > 0 else 1.0,
        "memory_ratio_dora_vs_lora": dora_mem / lora_mem if lora_mem > 0 else 1.0
    }
