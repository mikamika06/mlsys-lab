import ref


def check(workdir):
    from adapter.dora import dora_extra_parameters

    cases = ref.get_test_cases()
    matched = 0
    for c in cases:
        got = dora_extra_parameters(c["d_in"], c["d_out"], c["r"])
        if got == c["extra_params"]:
            matched += 1
    ok = 1.0 if matched == len(cases) else 0.0
    out = {"counts_matched": ok}
    return out
