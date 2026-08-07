import ref


def check(workdir):
    from memtrack.stats import parse_vm_stat

    out = {"stats_matched": 0.0}
    got = parse_vm_stat(ref.SAMPLE_VM_STAT)
    want = ref.parse_vm_stat(ref.SAMPLE_VM_STAT)
    if got == want and len(got) > 0:
        out["stats_matched"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
