import ref


def check(workdir):
    from kquant.metrics import calculate_kquant_bpw

    out = {"bpw_matched": 0.0}
    types = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]

    ok = 0
    for q_type in types:
        want = ref.calculate_kquant_bpw_ref(q_type)
        got = calculate_kquant_bpw(q_type)
        if abs(want - got) < 1e-5:
            ok += 1

    if ok == len(types):
        out["bpw_matched"] = 1.0
    return out
