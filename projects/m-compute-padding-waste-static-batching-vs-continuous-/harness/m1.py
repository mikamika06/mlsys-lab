import ref


def check(workdir):
    from batching.waste import compute_padding_waste

    requests = ref.generate_workload(num_requests=30, seed=123)
    max_batch_size = 4

    total_useful = sum(r["prompt_len"] + r["decode_len"] for r in requests)
    static_padded_total = 0
    for i in range(0, len(requests), max_batch_size):
        chunk = requests[i : i + max_batch_size]
        max_prompt = max(r["prompt_len"] for r in chunk)
        max_decode = max(r["decode_len"] for r in chunk)
        static_padded_total += len(chunk) * (max_prompt + max_decode)

    expected_static_waste = static_padded_total - total_useful
    expected_static_ratio = float(expected_static_waste) / float(static_padded_total)

    got = compute_padding_waste(requests, max_batch_size)

    out = {"waste_metrics_matched": 0.0}

    rel_err_static = abs(got.get("static_padded_tokens", -1) - expected_static_waste)
    rel_err_ratio = abs(got.get("static_waste_ratio", -1.0) - expected_static_ratio)
    useful_match = got.get("static_useful_tokens", -1) == total_useful
    cont_zero = got.get("continuous_padded_tokens", -1) == 0

    if rel_err_static == 0 and rel_err_ratio < 1e-5 and useful_match and cont_zero:
        out["waste_metrics_matched"] = 1.0
    else:
        out["_note"] = f"Expected static waste {expected_static_waste}, got {got}"

    return out
