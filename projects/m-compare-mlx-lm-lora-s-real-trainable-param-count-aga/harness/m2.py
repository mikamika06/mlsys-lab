import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness.ref as ref


def check(workdir):
    out = {"fuse_matched": 0.0, "bench_matched": 0.0}
    try:
        from mlx_lora_audit.bench import compare_lora_dora_metrics
        from mlx_lora_audit.fuse import verify_fusion

        fuse_data = ref.generate_fuse_fixture(100)

        res_lora = verify_fusion(
            fuse_data["base"],
            fuse_data["lora_a"],
            fuse_data["lora_b"],
            fuse_data["scale"],
            fuse_data["fused_lora"],
            use_dora=False,
        )

        res_dora = verify_fusion(
            fuse_data["base"],
            fuse_data["lora_a"],
            fuse_data["lora_b"],
            fuse_data["scale"],
            fuse_data["fused_dora"],
            use_dora=True,
            magnitude_vector=fuse_data["m_vec"],
        )

        fuse_ok = res_lora["is_equivalent"] and res_dora["is_equivalent"]
        if fuse_ok:
            out["fuse_matched"] = 1.0

        model_cfg = {"in_features": 64, "out_features": 128}
        batch_cfg = {"seq_len": 32, "batch_size": 2, "steps": 5}
        bench_res = compare_lora_dora_metrics(model_cfg, batch_cfg, r=4)

        has_keys = all(k in bench_res for k in ("lora", "dora", "time_ratio_dora_vs_lora", "memory_ratio_dora_vs_lora"))
        if has_keys and bench_res["dora"]["avg_memory_bytes"] > bench_res["lora"]["avg_memory_bytes"]:
            out["bench_matched"] = 1.0

    except Exception as e:  # noqa: BLE001
        out["_note"] = f"Error in m2 check: {type(e).__name__}: {str(e)}"
    return out
