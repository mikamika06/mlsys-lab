import ref
import torch

def check(workdir):
    from dynamotrace.rewrite import measure_latency, rewrite_fn

    args = ref.get_sample_args()
    try:
        lat_orig = measure_latency(torch.compile(ref.sample_target_fn), args, num_iters=20)
        lat_rewritten = measure_latency(torch.compile(rewrite_fn(ref.sample_target_fn)), args, num_iters=20)
    except Exception as e:
        return {"latency_ratio_valid": 0.0, "rewrite_match": 0.0, "_note": f"error: {type(e).__name__}: {str(e)[:120]}"}

    ratio_valid = 1.0 if lat_orig > 0 and lat_rewritten > 0 else 0.0

    rewritten_res = rewrite_fn(ref.sample_target_fn)(*args)
    expected_res = ref.sample_target_fn(*args)

    match_rewrite = 1.0 if torch.allclose(rewritten_res, expected_res) else 0.0

    return {
        "latency_ratio_valid": ratio_valid,
        "rewrite_match": match_rewrite
    }
