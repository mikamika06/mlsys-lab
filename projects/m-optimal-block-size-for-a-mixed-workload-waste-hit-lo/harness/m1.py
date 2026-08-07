import ref


def check(workdir):
    from kvblock.objective import (
        calculate_internal_fragmentation,
        calculate_prefix_truncation_loss,
        evaluate_workload_objective,
    )

    out = {"objective_matched": 0.0, "truncation_matched": 0.0}

    obj_ok = True
    trunc_ok = True

    for b_size in ref.CANDIDATE_BLOCK_SIZES:
        want_frag = ref.ref_calculate_internal_fragmentation(ref.SAMPLE_SEQS, b_size)
        got_frag = calculate_internal_fragmentation(ref.SAMPLE_SEQS, b_size)
        if want_frag != got_frag:
            obj_ok = False
            out["_note"] = f"frag mismatch at block_size={b_size}: got {got_frag}, want {want_frag}"
            break

        want_trunc = ref.ref_calculate_prefix_truncation_loss(
            ref.SHARED_PREFIXES, ref.REQUEST_PREFIXES, b_size
        )
        got_trunc = calculate_prefix_truncation_loss(
            ref.SHARED_PREFIXES, ref.REQUEST_PREFIXES, b_size
        )
        if want_trunc != got_trunc:
            trunc_ok = False
            out["_note"] = f"truncation mismatch at block_size={b_size}: got {got_trunc}, want {want_trunc}"
            break

        want_obj = ref.ref_evaluate_workload_objective(
            ref.SAMPLE_SEQS,
            ref.SHARED_PREFIXES,
            ref.REQUEST_PREFIXES,
            b_size,
            ref.HIT_PENALTY_WEIGHT,
        )
        got_obj = evaluate_workload_objective(
            ref.SAMPLE_SEQS,
            ref.SHARED_PREFIXES,
            ref.REQUEST_PREFIXES,
            b_size,
            ref.HIT_PENALTY_WEIGHT,
        )
        if abs(want_obj - got_obj) > 1e-5:
            obj_ok = False
            out["_note"] = f"objective mismatch at block_size={b_size}: got {got_obj}, want {want_obj}"
            break

    if obj_ok:
        out["objective_matched"] = 1.0
    if trunc_ok:
        out["truncation_matched"] = 1.0

    return out
