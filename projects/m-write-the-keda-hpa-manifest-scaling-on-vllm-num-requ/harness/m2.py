import ref


def check(workdir):
    from keda.manifest import generate_scaled_object
    from keda.validator import validate_scaled_object

    out = {"manifest_match": 0.0, "metadata_match": 0.0}
    ok_manifest = 0
    ok_meta = 0

    for cfg in ref.CONFIGS:
        want = ref.build_manifest(cfg)
        got = generate_scaled_object(cfg)
        if got == want:
            ok_manifest += 1
        if validate_scaled_object(got):
            ok_meta += 1

    out["manifest_match"] = 1.0 if ok_manifest == len(ref.CONFIGS) else 0.0
    out["metadata_match"] = 1.0 if ok_meta == len(ref.CONFIGS) else 0.0
    return out
