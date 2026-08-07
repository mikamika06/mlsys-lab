import ref


def check(workdir):
    from loraserving.routing import apply_per_row_lora, verify_batched_lora

    out = {"batches_verified": 0.0, "total_batches": float(len(ref.BATCH_CASES))}
    ok = 0
    for i, case in enumerate(ref.BATCH_CASES):
        got_out = apply_per_row_lora(
            case["x"], case["adapter_ids"], case["lora_a"], case["lora_b"], case["scaling"]
        )
        ver = verify_batched_lora(
            case["x"], case["adapter_ids"], case["lora_a"], case["lora_b"], case["scaling"], case["expected_out"]
        )
        if ver.get("is_correct", False) and ver.get("max_error", 1.0) < 1e-4:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"batch {i}: max_error {ver.get('max_error')}"

    out["batches_verified"] = float(ok)
    return out
