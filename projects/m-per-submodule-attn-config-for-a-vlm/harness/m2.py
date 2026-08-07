import ref


def check(workdir):
    from vlmattn.memory import compute_submodule_bytes
    from vlmattn.config import parse_submodule_configs

    out = {"bytes_match": 0.0, "submodule_match": 0.0}
    ok_bytes = 0
    ok_sub = 0

    for i, cfg in enumerate(ref.CONFIGS):
        want_bytes = ref.compute_submodule_bytes(cfg, 2, 1024)
        got_bytes = compute_submodule_bytes(cfg, 2, 1024)
        if got_bytes == want_bytes:
            ok_bytes += 1

        want_parsed = ref.parse_submodule_configs(cfg)
        got_parsed = parse_submodule_configs(cfg)
        if len(got_parsed) == len(want_parsed):
            ok_sub += 1

    out["bytes_match"] = 1.0 if ok_bytes == len(ref.CONFIGS) else 0.0
    out["submodule_match"] = 1.0 if ok_sub == len(ref.CONFIGS) else 0.0
    return out
