import math
import ref


def check(workdir):
    from block_sensitivity.workload import compare_workload_sensitivity, explain_hit_rate_regression

    chat_lengths = ref.get_chat_prompts(42)
    rag_prompts = ref.get_rag_prompts(42)
    block_sizes = [8, 16, 32, 64]

    out = {"sensitivity_matched": 0.0, "hit_rate_matched": 0.0}

    try:
        got_sens = compare_workload_sensitivity(chat_lengths, rag_prompts, block_sizes)
        want_sens = ref.oracle_sensitivity(chat_lengths, rag_prompts, block_sizes)

        valid = True
        for bs in block_sizes:
            if bs not in got_sens:
                valid = False
                break
            g = got_sens[bs]
            w = want_sens[bs]
            if (
                g.get("chat_total_blocks") != w["chat_total_blocks"]
                or g.get("rag_total_blocks") != w["rag_total_blocks"]
                or not math.isclose(g.get("chat_frag_rate", -1.0), w["chat_frag_rate"], rel_tol=1e-3)
                or not math.isclose(g.get("rag_frag_rate", -1.0), w["rag_frag_rate"], rel_tol=1e-3)
            ):
                valid = False
                break
        if valid:
            out["sensitivity_matched"] = 1.0
        else:
            out["_note_sens"] = f"compare_workload_sensitivity mismatch"
    except Exception as e:
        out["_note_sens_err"] = f"compare_workload_sensitivity raised {e}"

    try:
        prefix_len = 1050
        suffixes = [100, 120, 90, 110]
        bs1, bs2 = 16, 32

        got_explain = explain_hit_rate_regression(prefix_len, suffixes, bs1, bs2)
        want_explain = ref.oracle_explain(prefix_len, suffixes, bs1, bs2)

        if (
            got_explain.get("cached_tokens_bs1") == want_explain["cached_tokens_bs1"]
            and got_explain.get("cached_tokens_bs2") == want_explain["cached_tokens_bs2"]
            and math.isclose(got_explain.get("hit_rate_bs1", -1.0), want_explain["hit_rate_bs1"], rel_tol=1e-3)
            and math.isclose(got_explain.get("hit_rate_bs2", -1.0), want_explain["hit_rate_bs2"], rel_tol=1e-3)
            and math.isclose(got_explain.get("hit_rate_drop", -1.0), want_explain["hit_rate_drop"], rel_tol=1e-3)
        ):
            out["hit_rate_matched"] = 1.0
        else:
            out["_note_explain"] = f"explain_hit_rate_regression mismatch: got {got_explain}, want {want_explain}"
    except Exception as e:
        out["_note_explain_err"] = f"explain_hit_rate_regression raised {e}"

    return out
