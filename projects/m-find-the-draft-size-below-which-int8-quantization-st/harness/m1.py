import math
import ref

def check(workdir):
    from speculative_quant.throughput import expected_tokens, calculate_throughput

    out = {"et_matched": 0.0, "tp_matched": 0.0}
    et_ok = 0
    tp_ok = 0
    total_et = 0
    total_tp = 0

    for sc in ref.SCENARIOS:
        for alpha in sc["alphas_fp16"].values():
            total_et += 1
            want = ref.expected_tokens(alpha, sc["K"])
            got = expected_tokens(alpha, sc["K"])
            if math.isclose(want, got, rel_tol=1e-5):
                et_ok += 1
            elif "_note" not in out:
                out["_note"] = f"expected_tokens({alpha}, {sc['K']}) = {got}, want {want}"

        for s in sc["draft_sizes"]:
            for is_int8 in [False, True]:
                alpha = sc["alphas_int8"][s] if is_int8 else sc["alphas_fp16"][s]
                total_tp += 1
                want = ref.calculate_throughput(s, is_int8, sc["s_target"], sc["K"], sc["mem_bw"], alpha, sc["overheads"])
                got = calculate_throughput(s, is_int8, sc["s_target"], sc["K"], sc["mem_bw"], alpha, sc["overheads"])
                if math.isclose(want, got, rel_tol=1e-5):
                    tp_ok += 1
                elif "_note" not in out:
                    out["_note"] = f"calculate_throughput for size {s}, int8={is_int8} = {got}, want {want}"

    out["et_matched"] = et_ok / total_et
    out["tp_matched"] = tp_ok / total_tp
    return out
