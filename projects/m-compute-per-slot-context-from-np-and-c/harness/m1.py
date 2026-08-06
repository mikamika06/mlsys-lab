import sys
import harness.ref as ref


def check(workdir):
    sys.path.insert(0, workdir)
    from slotplan.context import compute_slot_context, validate_slot_allocation

    out = {"configs_matched": 0.0}
    cases = ref.generate_context_cases()
    ok = 0

    for case in cases:
        c_tot = case["c_total"]
        np_s = case["np_slots"]
        req = case["req"]

        want_ctx = c_tot // np_s
        got_ctx = compute_slot_context(c_tot, np_s)

        want_val = want_ctx >= req
        got_val = validate_slot_allocation(c_tot, np_s, req)

        if got_ctx == want_ctx and got_val == want_val:
            ok += 1

    if ok == len(cases):
        out["configs_matched"] = 1.0

    return out
