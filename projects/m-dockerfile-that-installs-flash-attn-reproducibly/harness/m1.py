import ref

def check(workdir):
    from flashbuild.dockerfile import generate_dockerfile
    out = {"dockerfile_valid": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.build_dockerfile(cfg)
        got = generate_dockerfile(cfg)
        if got and got.strip() == want.strip():
            ok += 1
    if ok == len(ref.CONFIGS):
        out["dockerfile_valid"] = 1.0
    return out
