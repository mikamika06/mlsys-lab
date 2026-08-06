import sys
import harness.ref as ref


def check(workdir):
    sys.path.insert(0, workdir)
    from slotplan.saturation import find_saturation_point

    out = {"saturation_matched": 0.0}
    cases = ref.generate_saturation_cases()
    ok = 0

    for case in cases:
        base = case["model_bytes_base"]
        overhead = case["slot_overhead_bytes"]
        max_mem = case["max_memory_bytes"]
        c_tot = case["c_total"]

        avail = max_mem - base
        per_slot = overhead * c_tot
        want_sat = avail // per_slot

        got_sat = find_saturation_point(base, overhead, max_mem, c_tot)

        if got_sat == want_sat:
            ok += 1

    if ok == len(cases):
        out["saturation_matched"] = 1.0

    return out
