import ref


def check(workdir):
    from vllm_diag.diagnosis import identify_zero_hit
    deployments = ref.get_deployments()
    want = ref.identify_zero_hit_deployment(deployments)
    got = identify_zero_hit(deployments)
    out = {"deployment_matched": 1.0 if got == want else 0.0}
    if got != want:
        out["_note"] = f"expected zero-hit deployment ID {want}, got {got}"
    return out
