import ref


def check(workdir):
    from preempt.recompute import preempt_recompute

    out = {"recompute_matched": 0.0, "tokens_rel_err": 1.0}
    all_matched = True
    total_ref_tokens = 0
    total_got_tokens = 0

    for item in ref.REQUEST_WORKLOADS:
        reqs = item["requests"]
        p_ids = item["preempt_ids"]

        want_reqs, want_tok = ref.preempt_recompute(reqs, p_ids)
        got_reqs, got_tok = preempt_recompute(reqs, p_ids)

        total_ref_tokens += want_tok
        total_got_tokens += got_tok

        if len(want_reqs) != len(got_reqs):
            all_matched = False
            break

        for w, g in zip(want_reqs, got_reqs):
            if w["req_id"] != g["req_id"] or w["status"] != g["status"] or w["num_blocks"] != g["num_blocks"]:
                all_matched = False
                break

    if all_matched:
        out["recompute_matched"] = 1.0

    denom = max(1.0, float(total_ref_tokens))
    out["tokens_rel_err"] = abs(float(total_got_tokens) - float(total_ref_tokens)) / denom
    return out
