import sys
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from cacheplan.layout import build_turn_prompt, total_processed_blocks

    sys_b = ref.SYS_BLOCKS
    hist_b = ref.HIST_STATES[5]
    dyn_b = ref.DYN_TURNS[5]

    got_p = build_turn_prompt(sys_b, hist_b, dyn_b)
    want_p = ref.build_turn_prompt(sys_b, hist_b, dyn_b)

    got_total = total_processed_blocks(ref.GOOD_LOG)
    want_total = ref.total_processed_blocks(ref.GOOD_LOG)

    rel_err = abs(got_total - want_total) / max(want_total, 1)

    return {
        "prompt_match": 1.0 if got_p == want_p else 0.0,
        "total_rel_err": float(rel_err)
    }
