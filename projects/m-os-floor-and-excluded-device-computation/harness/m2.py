import ref


def check(workdir):
    from edgeexport.convert import convert_variant_manifest

    out = {"conversions_matched": 0.0, "bit_exact": 0.0}
    matched = 0
    bit_exact_ok = True

    for i, raw_m in enumerate(ref.RAW_MANIFESTS):
        want = ref.convert_variant_manifest(raw_m)
        got = convert_variant_manifest(raw_m)

        if (
            isinstance(got, dict)
            and got.get("digest") == want["digest"]
            and got.get("manifest_bytes") == want["manifest_bytes"]
        ):
            matched += 1
        else:
            bit_exact_ok = False
            if "_note" not in out:
                out["_note"] = f"manifest {i}: mismatch in converted output"

    m_reordered = {
        "metadata": {"precision": "fp16", "author": "ml-team"},
        "version": 2,
        "layers": [
            {"name": "dense1", "weight": [0.12345678, -0.98765432]},
            {"name": "embed", "weight": [0.5, 0.25]},
        ],
        "name": "model_alpha",
    }
    want_alpha = ref.convert_variant_manifest(ref.RAW_MANIFESTS[0])
    got_alpha = convert_variant_manifest(m_reordered)
    if (
        not isinstance(got_alpha, dict)
        or got_alpha.get("digest") != want_alpha["digest"]
    ):
        bit_exact_ok = False

    out["conversions_matched"] = float(matched)
    out["bit_exact"] = (
        1.0 if bit_exact_ok and matched == len(ref.RAW_MANIFESTS) else 0.0
    )
    return out
