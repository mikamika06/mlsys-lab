import ref


def check(workdir):
    from benchaudit.detector import audit_benchmark_configs

    want = ref.audit_benchmark_configs(ref.AUDIT_CONFIGS)
    got = audit_benchmark_configs(ref.AUDIT_CONFIGS)

    if got == want:
        return {"configs_audited": 1.0}
    return {"configs_audited": 0.0, "_note": f"Expected {want}, got {got}"}
