import ref


def check(workdir):
    from distillcache.memory import (
        compute_full_vocab_footprint,
        compute_topk_footprint,
        max_samples_within_budget,
    )

    out = {
        "footprints_matched": 0.0,
        "topk_matched": 0.0,
        "ram_budget_matched": 0.0,
    }

    full_ok = 0
    for tc in ref.TEST_CASES_MEMORY:
        want = ref.ref_compute_full_vocab_footprint(**tc)
        got = compute_full_vocab_footprint(**tc)
        if got == want:
            full_ok += 1
    if full_ok == len(ref.TEST_CASES_MEMORY):
        out["footprints_matched"] = 1.0

    topk_ok = 0
    for tc in ref.TEST_CASES_TOPK:
        want = ref.ref_compute_topk_footprint(
            tc["num_samples"], tc["seq_len"], tc["top_k"], tc["logit_bytes"], tc["idx_bytes"]
        )
        got = compute_topk_footprint(
            tc["num_samples"], tc["seq_len"], tc["top_k"], tc["logit_bytes"], tc["idx_bytes"]
        )
        if got == want:
            topk_ok += 1
    if topk_ok == len(ref.TEST_CASES_TOPK):
        out["topk_matched"] = 1.0

    budget_ok = 0
    for tc in ref.TEST_CASES_BUDGET:
        want = ref.ref_max_samples_within_budget(**tc)
        got = max_samples_within_budget(**tc)
        if got == want:
            budget_ok += 1
    if budget_ok == len(ref.TEST_CASES_BUDGET):
        out["ram_budget_matched"] = 1.0

    return out
